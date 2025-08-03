import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .positional_encoding import RoPE

EPS = 1e-6
INF = 1e9


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism"""
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple:
        """
        Args:
            Q: Query tensor [batch, heads, seq_len, dim]
            K: Key tensor [batch, heads, seq_len, dim]
            V: Value tensor [batch, heads, seq_len, dim]
            mask: Attention mask [batch, heads, seq_len, seq_len]
        """
        # Compute attention scores
        attn = torch.matmul(Q / self.scale, K.transpose(2, 3))
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -INF)
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        
        return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE positional encoding"""
    
    def __init__(self, n_head: int, q_k_dim: int, v_dim: int, d_model: int):
        super().__init__()
        
        self.n_head = n_head
        self.q_k_dim = q_k_dim
        self.v_dim = v_dim
        
        # Linear projections
        self.Q_weight = nn.Linear(d_model, n_head * q_k_dim, bias=False)
        self.K_weight = nn.Linear(d_model, n_head * q_k_dim, bias=False)
        self.V_weight = nn.Linear(d_model, n_head * v_dim, bias=False)
        self.out_weight = nn.Linear(n_head * v_dim, d_model, bias=False)
        
        # Attention and positional encoding
        self.attention = ScaledDotProductAttention(q_k_dim ** 0.5)
        self.pe = RoPE(q_k_dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of multi-head attention"""
        batch_n, q_l, k_v_l = q.shape[0], q.shape[1], k.shape[1]
        
        # Linear projections and reshape
        Q = self.Q_weight(q).view(batch_n, q_l, self.n_head, self.q_k_dim)
        K = self.K_weight(k).view(batch_n, k_v_l, self.n_head, self.q_k_dim)
        V = self.V_weight(v).view(batch_n, k_v_l, self.n_head, self.v_dim)
        
        # Apply positional encoding
        if not hasattr(self, 'pe'):
            self.pe = RoPE(self.q_k_dim)
        Q, K = self.pe(Q), self.pe(K)
        
        # Transpose for attention computation
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Expand mask for multi-head
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # Compute attention
        out_Q, attn = self.attention(Q, K, V, mask)
        
        # Reshape and apply output projection
        out_Q = out_Q.transpose(1, 2).contiguous().view(batch_n, q_l, -1)
        out_Q = self.out_weight(out_Q)
        
        return out_Q


# Import moved to avoid circular dependency