import argparse
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any

from .utils.config import Config
from .utils.config_loader import load_config_from_yaml
from .utils.logger import get_logger
from .utils.registry import ModelRegistry, DatasetRegistry, TaskRegistry
from .models.lipm import LiPMModel
from .data.dataset import BatteryDataset
from .data.loader import BatteryDataLoader
from .training.trainer import BatteryTrainer
from .tasks.soh import SOHPredictor
from .tasks.soc import SOCPredictor

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Battery Foundation Model Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LiPM model with default configuration
  python -m battery_foundation.cli train --config configs/default.yaml
  
  # Train with custom data path
  python -m battery_foundation.cli train --config configs/default.yaml --data-path /path/to/data
  
  # Train with LODO (Leave-One-Domain-Out) - HUST as test set
  python -m battery_foundation.cli train --config configs/default.yaml --lodo HUST
  
  # Fine-tune for SOH prediction
  python -m battery_foundation.cli finetune --config configs/soh.yaml --checkpoint checkpoints/best_model.pth
  
  # Evaluate model
  python -m battery_foundation.cli evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pth
        """
    )
    
    # Common arguments
    parser.add_argument('command', choices=['train', 'finetune', 'evaluate', 'predict'],
                       help='Command to execute')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--data-path', type=str,
                       help='Override data path from config')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                       default='auto', help='Device to use for computation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Training specific arguments
    parser.add_argument('--epochs', type=int,
                       help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size')
    parser.add_argument('--lr', type=float,
                       help='Override learning rate')
    parser.add_argument('--lodo', type=str,
                       help='Leave-One-Domain-Out: specify dataset name for test set (e.g., HUST)')
    
    # Evaluation specific arguments
    parser.add_argument('--eval-split', type=str, choices=['train', 'val', 'test'],
                       default='test', help='Dataset split to evaluate on')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to file')
    
    return parser


def setup_config(args: argparse.Namespace) -> Config:
    """Setup configuration from arguments"""
    # Load base config
    config = load_config_from_yaml(args.config)
    
    # Override with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.max_epoch = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    
    # Handle LODO configuration
    if args.lodo:
        config.lodo_dataset = args.lodo
        logger.info(f"LODO mode enabled: {args.lodo} will be used as test dataset")
    
    # Setup device
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device
    
    # Set seed
    config.seed = args.seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    return config


def create_model(config: Config) -> torch.nn.Module:
    """Create model from configuration"""
    model_name = getattr(config, 'model_name', 'lipm')
    
    if model_name not in ModelRegistry:
        available_models = ModelRegistry.list_available()
        raise ValueError(f"Model '{model_name}' not found. Available: {available_models}")
    
    model_class = ModelRegistry.get(model_name)
    model = model_class(config)
    
    logger.info(f"Created {model_name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def create_dataset(config: Config, split: str = 'train') -> BatteryDataset:
    """Create dataset from configuration"""
    dataset_name = getattr(config, 'dataset_name', 'battery').lower()
    
    if dataset_name not in DatasetRegistry:
        available_datasets = DatasetRegistry.list_available()
        logger.warning(f"Dataset '{dataset_name}' not found. Available: {available_datasets}. Using 'battery'")
        dataset_name = 'battery'
    
    dataset_class = DatasetRegistry.get(dataset_name)
    
    # Get datasets list from config
    datasets_to_load = getattr(config, 'datasets', ['NASA'])
    
    # Handle LODO: filter out the test dataset from training/validation
    if hasattr(config, 'lodo_dataset') and config.lodo_dataset and split != 'test':
        lodo_dataset = config.lodo_dataset.upper()
        if lodo_dataset in datasets_to_load:
            datasets_to_load = [d for d in datasets_to_load if d != lodo_dataset]
            logger.info(f"LODO: Excluded {lodo_dataset} from {split} set. Remaining datasets: {datasets_to_load}")
        else:
            logger.warning(f"LODO dataset '{lodo_dataset}' not found in available datasets: {datasets_to_load}")
    
    # For test split in LODO mode, only load the specified dataset
    if hasattr(config, 'lodo_dataset') and config.lodo_dataset and split == 'test':
        lodo_dataset = config.lodo_dataset.upper()
        datasets_to_load = [lodo_dataset]
        logger.info(f"LODO: Using {lodo_dataset} as test dataset")
    
    # Create dataset with config parameters
    dataset_kwargs = {
        'data_path': config.data_path,
        'patch_len': config.patch_len,
        'patch_num': config.patch_num,
        'patch_stride': config.patch_stride,
        'n_var': config.n_var,
        'normalize': getattr(config, 'normalize', True)
    }
    
    # Only pass dataset_names for 'battery' dataset (multi-dataset loader)
    if dataset_name == 'battery':
        dataset_kwargs['dataset_names'] = datasets_to_load
    
    dataset = dataset_class(**dataset_kwargs)
    
    logger.info(f"Created {dataset_name} dataset for {split} with {len(dataset)} samples from {datasets_to_load}")
    
    return dataset


def train_command(args: argparse.Namespace, config: Config):
    """Execute training command"""
    logger.info("Starting training...")
    
    # Create model
    model = create_model(config)
    
    # Create datasets
    train_dataset = create_dataset(config, 'train')
    val_dataset = None  # TODO: Implement train/val split
    
    # Create test dataset for LODO mode
    test_dataset = None
    if hasattr(config, 'lodo_dataset') and config.lodo_dataset:
        test_dataset = create_dataset(config, 'test')
        logger.info(f"LODO mode: Created test dataset with {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = BatteryDataLoader(train_dataset, config, shuffle=True)
    val_loader = BatteryDataLoader(val_dataset, config, shuffle=False) if val_dataset else None
    test_loader = BatteryDataLoader(test_dataset, config, shuffle=False) if test_dataset else None
    
    # Create trainer
    trainer = BatteryTrainer(model, config, train_loader, val_loader, test_loader)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        logger.info(f"Resumed training from {args.checkpoint}")
    
    # Start training
    history = trainer.train()
    
    logger.info("Training completed successfully!")
    
    # Save training history
    if config.output_dir:
        import json
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
            # Convert any tensors to lists for JSON serialization
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        serializable_history[key] = [{k: float(v) for k, v in item.items()} for item in value]
                    else:
                        serializable_history[key] = [float(v) for v in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)


def finetune_command(args: argparse.Namespace, config: Config):
    """Execute fine-tuning command"""
    logger.info("Starting fine-tuning...")
    
    if not args.checkpoint:
        raise ValueError("Checkpoint required for fine-tuning")
    
    # Create base model
    base_model = create_model(config)
    
    # Load pretrained weights
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded pretrained model from {args.checkpoint}")
    
    # Create downstream task
    downstream_task = getattr(config, 'downstream_task', 'soh')
    if downstream_task not in TaskRegistry:
        available_tasks = TaskRegistry.list_available()
        raise ValueError(f"Task '{downstream_task}' not found. Available: {available_tasks}")
    
    task_class = TaskRegistry.get(downstream_task)
    model = task_class(base_model, config)
    
    # Create datasets and loaders
    train_dataset = create_dataset(config, 'train')
    val_dataset = None  # TODO: Implement train/val split
    
    train_loader = BatteryDataLoader(train_dataset, config, shuffle=True)
    val_loader = BatteryDataLoader(val_dataset, config, shuffle=False) if val_dataset else None
    
    # Create trainer
    trainer = BatteryTrainer(model, config, train_loader, val_loader)
    
    # Start fine-tuning
    history = trainer.train()
    
    logger.info("Fine-tuning completed successfully!")


def evaluate_command(args: argparse.Namespace, config: Config):
    """Execute evaluation command"""
    logger.info("Starting evaluation...")
    
    if not args.checkpoint:
        raise ValueError("Checkpoint required for evaluation")
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Create dataset
    dataset = create_dataset(config, args.eval_split)
    dataloader = BatteryDataLoader(dataset, config, shuffle=False)
    
    # Evaluate
    from .evaluation.metrics import BatteryMetrics
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = BatteryDataLoader.move_batch_to_device(batch, config.device)
            loss, loss_dict = model(batch)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f"Average loss on {args.eval_split} set: {avg_loss:.4f}")


def predict_command(args: argparse.Namespace, config: Config):
    """Execute prediction command"""
    logger.info("Starting prediction...")
    
    if not args.checkpoint:
        raise ValueError("Checkpoint required for prediction")
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Create dataset
    dataset = create_dataset(config, 'test')
    dataloader = BatteryDataLoader(dataset, config, shuffle=False)
    
    # Make predictions
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = BatteryDataLoader.move_batch_to_device(batch, config.device)
            
            # For downstream tasks, use predict method
            if hasattr(model, 'predict'):
                pred = model.predict(batch)
            else:
                pred = model(batch)
            
            predictions.append(pred.cpu().numpy())
    
    logger.info(f"Generated predictions for {len(predictions)} batches")
    
    # Save predictions if requested
    if args.save_predictions and config.output_dir:
        import numpy as np
        os.makedirs(config.output_dir, exist_ok=True)
        np.save(os.path.join(config.output_dir, 'predictions.npy'), 
                np.concatenate(predictions, axis=0))
        logger.info(f"Predictions saved to {config.output_dir}/predictions.npy")


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Setup configuration
        config = setup_config(args)
        
        logger.info(f"Battery Foundation Model CLI - Command: {args.command}")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Device: {config.device}")
        
        # Execute command
        if args.command == 'train':
            train_command(args, config)
        elif args.command == 'finetune':
            finetune_command(args, config)
        elif args.command == 'evaluate':
            evaluate_command(args, config)
        elif args.command == 'predict':
            predict_command(args, config)
        else:
            raise ValueError(f"Unknown command: {args.command}")
            
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()