import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, ScaledDotProductAttention
from .positional_encoding import IrregularRoPE

EPS = 1e-6


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """Feed-forward network with ReLU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with attention and feed-forward layers"""
    
    def __init__(self, config):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            config.n_head, 
            config.d_model // config.n_head,
            config.d_model // config.n_head, 
            config.d_model
        )
        self.feed_forward = FeedForward(
            config.d_model, 
            config.d_model * 4, 
            config.dp
        )
        
        # Normalization layers
        if config.norm == 'rsm':
            self.attention_norm = RMSNorm(config.d_model)
            self.ffn_norm = RMSNorm(config.d_model)
        else:
            self.attention_norm = nn.LayerNorm(config.d_model)
            self.ffn_norm = nn.LayerNorm(config.d_model)
        
        self.pre_norm = config.pre_norm
        self.dropout = nn.Dropout(config.dp)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, 
                k: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of transformer block"""
        
        # Self-attention with residual connection
        if self.pre_norm:
            q = self.attention_norm(x)
        else:
            q = x
        
        if k is None:
            k, v = q, q
        
        h = x + self.dropout(self.attention(q, k, v, mask))
        
        if not self.pre_norm:
            h = self.attention_norm(h)
        
        # Feed-forward with residual connection
        if self.pre_norm:
            _h = self.ffn_norm(h)
        else:
            _h = h
        
        out = h + self.feed_forward(_h)
        
        if not self.pre_norm:
            out = self.ffn_norm(out)
        
        return out


class iBlock(nn.Module):
    """Inverted Transformer Block for patch-wise processing with irregular time encoding"""
    
    def __init__(self, config):
        super().__init__()
        
        # Attention parameters
        self.n_head = 8
        self.q_k_dim = 32
        self.v_dim = 32
        
        # Linear projections for Q, K, V
        self.Q_weight = nn.Linear(config.n_var, self.n_head * self.q_k_dim, bias=False)
        self.K_weight = nn.Linear(config.n_var, self.n_head * self.q_k_dim, bias=False)
        self.V_weight = nn.Linear(config.n_var, self.n_head * self.v_dim, bias=False)
        
        # Output projection
        self.out_weight = nn.Linear(self.n_head * self.v_dim, config.d_model, bias=False)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.q_k_dim ** 0.5)
        self.pe = IrregularRoPE(self.q_k_dim)
        
        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(self.n_head * self.v_dim)
        self.ffn = FeedForward(self.n_head * self.v_dim, 4 * self.n_head * self.v_dim)
        
        # Learnable embedding token
        self.emb_token = nn.Parameter(torch.zeros((1, 1, 1, 2)))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of iBlock"""
        batch_n, pn, pl = x.shape[0], x.shape[1], x.shape[2]
        
        # Add embedding token
        emb_token = self.emb_token.expand(batch_n, pn, 1, -1)
        _x = torch.cat((emb_token, x), dim=2)  # [batch, patches, seq_len+1, vars]
        
        # Compute Q, K, V
        Q_out = self.Q_weight(_x)  # [batch_n, pn, pl+1, n_head*q_k_dim]
        K_out = self.K_weight(_x)  # [batch_n, pn, pl+1, n_head*q_k_dim] 
        V_out = self.V_weight(_x)  # [batch_n, pn, pl+1, n_head*v_dim]
        
        Q = Q_out.view(batch_n * pn, pl + 1, self.n_head, self.q_k_dim)
        K = K_out.view(batch_n * pn, pl + 1, self.n_head, self.q_k_dim)
        V = V_out.view(batch_n * pn, pl + 1, self.n_head, self.v_dim)
        
        # Apply positional encoding
        t = t.view(batch_n * pn, -1)
        Q, K = self.pe(Q, t), self.pe(K, t)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Prepare attention mask
        if mask is not None:
            if len(mask.shape) == 3:
                mask_mat = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            else:
                mask_mat = mask
            mask_mat = mask_mat.view(batch_n * pn, pl + 1, pl + 1)
            mask_mat = mask_mat.unsqueeze(1)
        else:
            mask_mat = None
        
        # Apply attention
        _Q, _ = self.attention(Q, K, V, mask_mat)
        
        # Extract first token (embedding token) output
        last_Q = _Q[:, :, 0, :]  # [batch*patches, heads, dim]
        last_Q = last_Q.contiguous().view(batch_n, pn, -1)
        
        # Apply feed-forward network
        last_Q = last_Q + self.ffn(self.ffn_norm(last_Q))
        
        # Final output projection
        out = self.out_weight(last_Q)
        
        return out


class iBlockRaw(nn.Module):
    """Raw iBlock for direct sequence processing"""
    
    def __init__(self, config, n_var: int = 1):
        super().__init__()
        
        self.n_head = 4
        self.q_k_dim = 4 * n_var
        self.v_dim = 4 * n_var
        self.n_var = n_var
        
        # Linear projections
        self.Q_weight = nn.Linear(n_var, self.n_head * self.q_k_dim, bias=False)
        self.K_weight = nn.Linear(n_var, self.n_head * self.q_k_dim, bias=False)
        self.V_weight = nn.Linear(n_var, self.n_head * self.v_dim, bias=False)
        
        # Output projection
        self.out_weight = nn.Linear(self.n_head * self.v_dim, n_var, bias=False)
        
        # Feed-forward network
        self.ffn = FeedForward(self.n_head * self.v_dim, self.n_head * self.v_dim * 4)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.q_k_dim ** 0.5)
        self.pe = IrregularRoPE(self.q_k_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, 
                attn: bool = False) -> torch.Tensor:
        """Forward pass of iBlockRaw"""
        
        # Handle variable dimensions
        if x.shape[-1] != self.n_var:
            if self.n_var == 1:
                x = x.unsqueeze(-1)
            else:
                raise Exception(f'n_var is {self.n_var}, but shape of x is {x.shape}')
        
        batch_n, pn, pl = x.shape[0], x.shape[1], x.shape[2]
        
        # Compute Q, K, V
        Q = self.Q_weight(x).view(batch_n * pn, pl, self.n_head, self.q_k_dim)
        K = self.K_weight(x).view(batch_n * pn, pl, self.n_head, self.q_k_dim)
        V = self.V_weight(x).view(batch_n * pn, pl, self.n_head, self.v_dim)
        
        # Apply positional encoding
        t = t.view(batch_n * pn, -1)
        Q, K = self.pe(Q, t), self.pe(K, t)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Prepare attention mask
        if mask is not None:
            if len(mask.shape) == 3:
                mask_a = mask.unsqueeze(-1)
                mask_b = mask.unsqueeze(-2)
                mask_mat = mask_a * mask_b
            else:
                mask_mat = mask
            mask_mat = mask_mat.view(batch_n * pn, pl, pl)
            mask_mat = mask_mat.unsqueeze(1)
        else:
            mask_mat = None
        
        # Apply attention
        _Q, _attn = self.attention(Q, K, V, mask_mat)
        _Q = _Q.transpose(1, 2).contiguous().view(batch_n, pn, pl, -1)
        
        # Apply feed-forward network
        out_Q = _Q + self.ffn(_Q)
        
        # Final output projection
        out = self.out_weight(out_Q)
        
        if attn:
            return out, _attn, _Q, out_Q, Q, K, V
        
        return out