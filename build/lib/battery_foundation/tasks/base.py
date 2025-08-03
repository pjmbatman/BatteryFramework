import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

from ..utils.config import Config
from ..utils.registry import TaskRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseDownstreamTask(nn.Module, ABC):
    """Base class for downstream battery analysis tasks"""
    
    def __init__(self, backbone_model: nn.Module, config: Config):
        """
        Initialize downstream task
        
        Args:
            backbone_model: Pretrained backbone model (e.g., LiPM)
            config: Configuration object
        """
        super().__init__()
        
        self.backbone = backbone_model
        self.config = config
        self.emb_dim = config.emb_dim
        
        # Freeze backbone if specified
        self.freeze_backbone = getattr(config, 'freeze_backbone', True)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone model frozen for downstream task")
        
        # Initialize task-specific components
        self._build_task_head()
        
        logger.info(f"Initialized {self.__class__.__name__} downstream task")
    
    @abstractmethod
    def _build_task_head(self):
        """Build task-specific head architecture"""
        pass
    
    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """Forward pass for the downstream task"""
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss"""
        pass
    
    def get_embeddings(self, batch) -> torch.Tensor:
        """Get embeddings from backbone model"""
        vc, qdqc, t, tm, pm = batch
        
        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.backbone.backbone(vc, t, tm, pm)
        else:
            embeddings = self.backbone.backbone(vc, t, tm, pm)
        
        return embeddings
    
    def predict(self, batch) -> torch.Tensor:
        """Make predictions for the downstream task"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)
        return predictions
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the task"""
        return {
            'task_name': self.__class__.__name__,
            'backbone_frozen': self.freeze_backbone,
            'embedding_dim': self.emb_dim,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }