import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base import BaseDownstreamTask
from ..utils.registry import TaskRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


@TaskRegistry.register("soh")
class SOHPredictor(BaseDownstreamTask):
    """State of Health (SOH) prediction downstream task"""
    
    def _build_task_head(self):
        """Build SOH prediction head"""
        # Get dimensions from config
        hidden_dim = getattr(self.config, 'soh_hidden_dim', 256)
        num_layers = getattr(self.config, 'soh_num_layers', 2)
        dropout = getattr(self.config, 'soh_dropout', 0.2)
        
        # Global pooling for sequence aggregation
        self.pooling_type = getattr(self.config, 'soh_pooling', 'mean')  # 'mean', 'max', 'last', 'attention'
        
        if self.pooling_type == 'attention':
            self.attention_pooling = AttentionPooling(self.emb_dim)
        
        # MLP head for regression
        layers = []
        input_dim = self.emb_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final prediction layer (SOH is between 0 and 1)
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1] range
        
        self.prediction_head = nn.Sequential(*layers)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        logger.info(f"SOH predictor with {num_layers} layers, {hidden_dim} hidden dim, {self.pooling_type} pooling")
    
    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass for SOH prediction
        
        Args:
            batch: Input batch (vc, qdqc, t, tm, pm)
            
        Returns:
            SOH predictions [batch_size, 1]
        """
        # Get embeddings from backbone
        embeddings = self.get_embeddings(batch)  # [batch, patches, emb_dim]
        
        # Aggregate patch embeddings
        if self.pooling_type == 'mean':
            aggregated = torch.mean(embeddings, dim=1)
        elif self.pooling_type == 'max':
            aggregated, _ = torch.max(embeddings, dim=1)
        elif self.pooling_type == 'last':
            aggregated = embeddings[:, -1, :]
        elif self.pooling_type == 'attention':
            aggregated = self.attention_pooling(embeddings)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        # Predict SOH
        soh_pred = self.prediction_head(aggregated)
        
        return soh_pred
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute SOH prediction loss"""
        return self.criterion(predictions.squeeze(), targets.squeeze())
    
    def predict_capacity_degradation(self, batch) -> Dict[str, torch.Tensor]:
        """
        Predict detailed capacity degradation information
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary with SOH predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            # Get embeddings
            embeddings = self.get_embeddings(batch)
            
            # Get patch-wise predictions for analysis
            patch_soh = []
            for i in range(embeddings.shape[1]):
                patch_emb = embeddings[:, i:i+1, :]  # Single patch
                if self.pooling_type == 'attention':
                    pooled = self.attention_pooling(patch_emb)
                else:
                    pooled = patch_emb.squeeze(1)
                patch_pred = self.prediction_head(pooled)
                patch_soh.append(patch_pred)
            
            patch_soh = torch.cat(patch_soh, dim=1)  # [batch, patches, 1]
            
            # Overall SOH prediction
            overall_soh = self.forward(batch)
            
            # Compute confidence as inverse of prediction variance
            confidence = 1.0 / (torch.var(patch_soh, dim=1) + 1e-6)
            
            return {
                'soh_prediction': overall_soh,
                'patch_soh': patch_soh,
                'confidence': confidence,
                'degradation_rate': torch.mean(torch.diff(patch_soh, dim=1), dim=1)
            }


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation"""
    
    def __init__(self, emb_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, emb_dim]
            
        Returns:
            Pooled tensor [batch, emb_dim]
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        pooled = torch.sum(x * attn_weights, dim=1)  # [batch, emb_dim]
        
        return pooled


@TaskRegistry.register("capacity_prediction")
class CapacityPredictor(SOHPredictor):
    """Variant of SOH predictor for direct capacity prediction"""
    
    def _build_task_head(self):
        """Build capacity prediction head (similar to SOH but different output range)"""
        super()._build_task_head()
        
        # Replace final sigmoid with linear layer for capacity values
        # Capacity can be in various ranges depending on battery type
        self.prediction_head[-1] = nn.Identity()  # Remove sigmoid
        
        # Add capacity range scaling if specified
        self.capacity_range = getattr(self.config, 'capacity_range', [0.0, 2.0])  # Ah
        
        logger.info(f"Capacity predictor with range {self.capacity_range}")
    
    def forward(self, batch) -> torch.Tensor:
        """Forward pass with capacity range scaling"""
        # Get base prediction
        pred = super().forward(batch)
        
        # Scale to capacity range
        min_cap, max_cap = self.capacity_range
        scaled_pred = pred * (max_cap - min_cap) + min_cap
        
        return scaled_pred