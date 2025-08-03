import os
import torch
import time
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..data.loader import BatteryDataLoader
from .loss_tracker import LossTracker
from .optimizer import get_optimizer
from .scheduler import get_scheduler

logger = get_logger(__name__)


class BatteryTrainer:
    """Professional trainer for battery foundation models"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 config: Config,
                 train_dataloader: BatteryDataLoader,
                 val_dataloader: Optional[BatteryDataLoader] = None):
        """
        Initialize trainer
        
        Args:
            model: The model to train
            config: Configuration object
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Setup loss tracking
        self.loss_tracker = LossTracker(['total', 'MMAE_mse', 'CIR_mse', 'MMAE_mae', 'CIR_mae'])
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = BatteryDataLoader.move_batch_to_device(batch, self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, loss_dict = self.model(batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track losses
            self.loss_tracker.update(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.current_step += 1
            
            # Check for early stopping based on max_iter
            if self.current_step >= self.config.max_iter:
                logger.info(f"Reached maximum iterations: {self.config.max_iter}")
                return self.loss_tracker.get_average()
        
        # Get epoch losses
        epoch_losses = self.loss_tracker.get_average()
        self.loss_tracker.reset()
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_loss_tracker = LossTracker(['total', 'MMAE_mse', 'CIR_mse', 'MMAE_mae', 'CIR_mae'])
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch = BatteryDataLoader.move_batch_to_device(batch, self.device)
                
                # Forward pass
                loss, loss_dict = self.model(batch)
                val_loss_tracker.update(loss_dict)
        
        return val_loss_tracker.get_average()
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.max_epoch} epochs or {self.config.max_iter} iterations")
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.config.max_epoch):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            training_history['train_losses'].append(train_losses)
            training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            training_history['epochs'].append(epoch)
            
            # Validation
            val_losses = self.validate()
            if val_losses:
                training_history['val_losses'].append(val_losses)
                logger.info(f"Epoch {epoch}: Train Loss: {train_losses['total']:.4f}, "
                           f"Val Loss: {val_losses['total']:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss: {train_losses['total']:.4f}")
            
            # Save checkpoint
            current_loss = val_losses.get('total', train_losses['total'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self._save_checkpoint(is_best=True)
                logger.info(f"New best model saved with loss: {self.best_loss:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(is_best=False)
            
            # Early stopping check
            if self.current_step >= self.config.max_iter:
                logger.info(f"Training stopped at epoch {epoch} due to max_iter reached")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final checkpoint
        self._save_checkpoint(is_best=False, filename="final_checkpoint.pth")
        
        return training_history
    
    def _save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """Save model checkpoint"""
        if filename is None:
            if is_best:
                filename = "best_model.pth"
            else:
                filename = f"checkpoint_epoch_{self.current_epoch}_step_{self.current_step}.pth"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            loss=self.best_loss,
            checkpoint_dir=self.config.checkpoint_dir,
            filename=filename
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint_info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.current_epoch = checkpoint_info['epoch']
        self.current_step = checkpoint_info['step']
        self.best_loss = checkpoint_info['loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.current_step}")
        
        return checkpoint_info
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': str(self.device),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_loss': self.best_loss
        }