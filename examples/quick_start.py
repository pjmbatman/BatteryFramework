"""
Quick start example for Battery Foundation Model Framework
"""

import os
import sys
import torch

# Add the parent directory to the path to import battery_foundation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from battery_foundation.utils.config_loader import load_config_from_yaml
from battery_foundation.models.lipm import LiPMModel
from battery_foundation.data.dataset import BatteryDataset
from battery_foundation.data.loader import BatteryDataLoader
from battery_foundation.training.trainer import BatteryTrainer


def main():
    """Quick start example"""
    print("🔋 Battery Foundation Model Framework - Quick Start")
    print("=" * 50)
    
    # 1. Load configuration
    config_path = "configs/small_model.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Please run this script from the BatteryFoundation directory")
        return
    
    print(f"📋 Loading configuration from {config_path}")
    config = load_config_from_yaml(config_path)
    
    # 2. Check data availability
    data_path = "../data/processed"
    if not os.path.exists(data_path):
        print(f"❌ Data directory not found: {data_path}")
        print("Please ensure battery data is available in the specified path")
        return
    
    print(f"📁 Data directory found: {data_path}")
    
    # 3. Create model
    print("🧠 Creating LiPM model...")
    model = LiPMModel(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 4. Create dataset and dataloader
    print("📊 Creating dataset...")
    try:
        dataset = BatteryDataset(
            data_path=config.data_path,
            dataset_names=['NASA'],  # Start with NASA dataset
            patch_len=config.patch_len,
            patch_num=config.patch_num,
            patch_stride=config.patch_stride,
            n_var=config.n_var,
            normalize=True
        )
        print(f"✅ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        print("Please check that the data is in the correct format")
        return
    
    # 5. Create dataloader
    print("🔄 Creating dataloader...")
    dataloader = BatteryDataLoader(dataset, config, shuffle=True, num_workers=0)
    print(f"✅ Dataloader created with batch size {config.batch_size}")
    
    # 6. Test a forward pass
    print("🔬 Testing forward pass...")
    try:
        model.eval()
        for batch in dataloader:
            with torch.no_grad():
                loss, loss_dict = model(batch)
                print(f"✅ Forward pass successful!")
                print(f"   Loss: {loss.item():.4f}")
                print(f"   Loss components: {loss_dict}")
                break
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return
    
    # 7. Setup trainer
    print("🏋️ Setting up trainer...")
    try:
        trainer = BatteryTrainer(model, config, dataloader)
        model_info = trainer.get_model_summary()
        print("✅ Trainer created successfully!")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"   Model size: {model_info['model_size_mb']:.1f} MB")
        print(f"   Device: {model_info['device']}")
    except Exception as e:
        print(f"❌ Error setting up trainer: {e}")
        return
    
    print("\n🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Run full training: battery-foundation train --config configs/small_model.yaml")
    print("2. Monitor training progress in the logs/ directory")
    print("3. Use trained model for downstream tasks (SOH, SOC)")
    print("4. Explore other configurations in configs/ directory")


if __name__ == "__main__":
    main()