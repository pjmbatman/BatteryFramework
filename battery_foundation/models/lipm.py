import torch
import torch.nn as nn
import numpy as np

from .transformer_blocks import TransformerBlock, iBlock, iBlockRaw, RMSNorm, FeedForward
from .attention import MultiHeadAttention
from ..utils.registry import ModelRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)

EPS = 1e-6


class iTransformer(nn.Module):
    """Inverted Transformer backbone for battery time series processing"""
    
    def __init__(self, config):
        super().__init__()
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            self.layers.append(TransformerBlock(config))
        
        # Input embedding and output projection
        self.to_embedding = iBlock(config)
        self.output = nn.Linear(config.d_model, config.emb_dim, bias=False)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                t_mask: torch.Tensor = None, patch_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of iTransformer
        
        Args:
            x: Input tensor [batch, patches, seq_len, vars]
            t: Time tensor [batch, patches, seq_len]
            t_mask: Time mask [batch, patches, seq_len]
            patch_mask: Patch mask [batch, patches]
        """
        # Convert to embeddings
        h = self.to_embedding(x.contiguous(), t, t_mask)
        
        # Apply transformer layers
        if patch_mask is not None:
            patch_mask_mat = patch_mask.unsqueeze(-1) * patch_mask.unsqueeze(-2)
        else:
            patch_mask_mat = None
            
        for layer in self.layers:
            h = layer(h, patch_mask_mat)
        
        # Output projection
        out = self.output(h)
        return out


class Seq2SeqHead(nn.Module):
    """Sequence-to-sequence head for reconstruction and capacity prediction"""
    
    def __init__(self, config, use_Q: bool = False, n_var: int = 2):
        super().__init__()
        
        # Key-Value merge layer
        self.kvs_merge = nn.Linear(config.emb_dim, config.down_dim)
        
        # Query parameters for capacity prediction
        if use_Q:
            self.Q = nn.Parameter(torch.zeros((config.patch_num, config.down_dim)))
        else:
            self.Q = None
        
        # Cross-attention layers
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(InterBlock(config))
        
        # Normalization and output projection
        self.norm = nn.Sequential(
            RMSNorm(config.down_dim),
            nn.Linear(config.down_dim, config.down_dim, bias=False)
        )
        self.lin_out = nn.Linear(config.down_dim, config.patch_len * n_var)
        
        # Output refinement layers
        self.out_layers = nn.ModuleList()
        for i in range(2):
            self.out_layers.append(iBlockRaw(config, n_var))
        
        self.n_var = n_var
    
    def forward(self, emb: torch.Tensor, t: torch.Tensor, t_m: torch.Tensor, 
                patch_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of Seq2Seq head
        
        Args:
            emb: Input embeddings [batch, patches, emb_dim]
            t: Time tensor [batch, patches, seq_len]
            t_m: Time mask [batch, patches, seq_len]
            patch_mask: Patch mask [batch, patches]
        """
        # Merge key-values
        kv = self.kvs_merge(emb)
        
        # Use learnable queries or key-values as queries
        Q = self.Q.expand(emb.shape[0], -1, -1) if self.Q is not None else kv
        
        # Prepare patch mask matrix
        if patch_mask is not None:
            patch_mask_mat = patch_mask.unsqueeze(-1) * patch_mask.unsqueeze(-2)
        else:
            patch_mask_mat = None
        
        # Apply cross-attention layers
        for layer in self.layers:
            Q = layer(Q, kv if self.Q is not None else Q, patch_mask_mat)
        
        # Apply normalization
        h = self.norm(Q)
        
        # Generate output
        out = self.lin_out(h).view(h.shape[0], h.shape[1], -1, self.n_var)
        
        # Apply output refinement layers
        if t_m is not None:
            mask_mat = t_m.unsqueeze(-1) * t_m.unsqueeze(-2)
        else:
            mask_mat = None
            
        for layer in self.out_layers:
            out = layer(out, t, mask_mat)
        
        return out


class InterBlock(nn.Module):
    """Interaction block for cross-attention between queries and key-values"""
    
    def __init__(self, config):
        super().__init__()
        
        down_dim = getattr(config, 'down_dim', 256)
        down_n_head = getattr(config, 'down_n_head', 4)
        
        # Multi-head attention layers
        self.self_attn = MultiHeadAttention(
            down_n_head, 
            down_dim // down_n_head, 
            down_dim // down_n_head, 
            down_dim
        )
        self.inter_attn = MultiHeadAttention(
            down_n_head, 
            down_dim // down_n_head, 
            down_dim // down_n_head, 
            down_dim
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(down_dim, down_dim * 4)
        
        # Normalization layers
        self.kv_norm = RMSNorm(down_dim)
        self.q_norm = RMSNorm(down_dim)
        self.ffn_norm = RMSNorm(down_dim)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of interaction block"""
        # Self-attention on queries
        _q = self.q_norm(q)
        out_q = q + self.self_attn(_q, _q, _q, mask)
        
        # Cross-attention between queries and key-values
        norm_q = self.q_norm(out_q)
        norm_kv = self.kv_norm(kv)
        h = out_q + self.inter_attn(norm_q, norm_kv, norm_kv, mask)
        
        # Feed-forward network
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out


@ModelRegistry.register("lipm")
class LiPMModel(nn.Module):
    """
    LiPM (Lithium-ion battery Performance Model) implementation
    
    This model implements the LiPM architecture for battery time series analysis
    with masked autoencoding pretraining and capacity prediction capabilities.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.backbone = iTransformer(config)
        
        # Initialize task-specific heads for pretraining
        if 'pretrain' in config.task:
            # Voltage-Current reconstruction head
            self.inter_head = Seq2SeqHead(config, use_Q=False, n_var=2)
            # Capacity prediction head  
            self.Q_head = Seq2SeqHead(config, use_Q=True, n_var=2)
            
            # Loss function
            self.mse_loss = nn.MSELoss()
            
            # Patching parameters
            self.patch_len = config.patch_len
            
            # Random masking state
            self.cur_idx_inter = 0
            self.idx_pool = np.arange(config.max_iter)
            np.random.shuffle(self.idx_pool)
        
        logger.info(f"Initialized LiPM model with task: {config.task}")
    
    def get_embedding(self, seq: torch.Tensor, t: torch.Tensor, 
                     tm: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        """Get embeddings without gradients for downstream tasks"""
        self.backbone.eval()
        with torch.no_grad():
            emb = self.backbone(seq, t, tm, pm)
        return emb.detach()
    
    def get_mask(self, x: torch.Tensor, pm: torch.Tensor, 
                tm: torch.Tensor) -> tuple:
        """Generate random masks for pretraining"""
        # Random patch and channel masking
        _patch_rand = torch.rand(x.shape[0], x.shape[1])
        _channel_rand = torch.rand(x.shape[0], x.shape[1], x.shape[3])
        
        # Create masks based on ratios
        mask_patch = (_patch_rand < self.config.patch_ratio).bool().to(x.device)
        mask_channel = (_channel_rand < self.config.channel_ratio).bool().to(x.device)
        
        # Forward mask (which patches to process)
        forward_mask = pm & ~mask_patch
        
        # Loss mask (which elements to reconstruct)
        loss_mask_patch = pm.unsqueeze(-1) & (mask_patch.unsqueeze(-1) | mask_channel)
        loss_mask = loss_mask_patch.unsqueeze(2) & tm.unsqueeze(3)
        
        # Apply masking to input
        masked_x = x * (~loss_mask)
        
        return forward_mask, loss_mask, masked_x
    
    def calculate_mae_loss(self, pred: torch.Tensor, y: torch.Tensor, 
                          loss_mask: torch.Tensor, t_mask: torch.Tensor) -> tuple:
        """Calculate masked autoencoder loss"""
        shape = y.shape
        pred = pred.view(shape[0], -1, shape[-1])
        y = y.view(shape[0], -1, shape[-1])
        loss_mask = loss_mask.view(shape[0], -1, shape[-1])
        
        # MSE loss
        loss = torch.mul(pred - y, loss_mask).pow(2).mean()
        
        # MAE for monitoring
        mae_loss = torch.mul(torch.abs(pred - y), loss_mask).mean()
        
        return loss, mae_loss.detach()
    
    def calculate_q_loss(self, pred: torch.Tensor, y: torch.Tensor, 
                        t_mask: torch.Tensor) -> tuple:
        """Calculate capacity prediction loss"""
        shape = y.shape
        pred = pred.view(shape[0], -1, shape[-1])
        y = y.view(shape[0], -1, shape[-1])
        t_mask = t_mask.reshape(shape[0], -1)
        
        # Weighted MSE loss
        loss = (torch.mul(pred - y, t_mask.unsqueeze(-1)).pow(2).sum() / 
                (t_mask.sum() * pred.shape[-1] + EPS))
        
        # Weighted MAE for monitoring
        mae_loss = (torch.mul(torch.abs(pred - y), t_mask.unsqueeze(-1)).sum() / 
                   (t_mask.sum() * pred.shape[-1] + EPS))
        
        return loss, mae_loss.detach()
    
    def forward(self, batch) -> tuple:
        """
        Forward pass for pretraining
        
        Args:
            batch: Tuple of (vc, QdQc, t, tm, pm)
                vc: Voltage-Current data [batch, patches, seq_len, 2]
                QdQc: Discharge-Charge capacity [batch, patches, 2]
                t: Time tensor [batch, patches+1]
                tm: Time mask [batch, patches+1, patches+1]
                pm: Patch mask [batch, patches]
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        vc, QdQc, t, tm, pm = batch
        
        # Generate masks for pretraining
        forward_mask, loss_mask, masked_vc = self.get_mask(vc, pm, tm)
        
        # Get embeddings from masked and full inputs
        emb_mask = self.backbone(masked_vc, t, tm, forward_mask)
        emb_full = self.backbone(vc, t, tm, pm)
        
        # Predict voltage-current and capacity
        pred_vc = self.inter_head(emb_mask, t, tm, forward_mask)
        pred_Q = self.Q_head(emb_full, t, tm, pm)
        
        # Calculate losses
        loss_MAE, MAE_mae = self.calculate_mae_loss(pred_vc, vc, loss_mask, tm)
        loss_Q, Q_mae = self.calculate_q_loss(pred_Q, QdQc, tm)
        
        # Combine losses
        loss = (self.config.weight_MAE * loss_MAE + self.config.weight_Q * loss_Q)
        loss = loss / (self.config.weight_MAE + self.config.weight_Q)
        
        # Prepare loss dictionary
        loss_dict = {
            'total': loss.item(),
            'MMAE_mse': loss_MAE.item(),
            'MMAE_mae': MAE_mae.item(),
            'CIR_mse': loss_Q.item(),
            'CIR_mae': Q_mae.item()
        }
        
        return loss, loss_dict