import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class RoPE(nn.Module):
    """Rotary Positional Embedding (RoPE) for regular sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.freqs_cis = self.precompute_freqs_cis(d_model, max_len)
    
    def precompute_freqs_cis(self, d_model: int, max_len: int, theta: float = 1000.0):
        """Precompute frequency components for RoPE"""
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[:(d_model//2)].float() / d_model))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcast frequency components to match input tensor shape"""
        dim_n = len(x.shape)
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == dim_n - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape).to(x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor"""
        with autocast(enabled=False):
            x_ = x.reshape(*x.shape[:-1], -1, 2).float()
            x_ = torch.view_as_complex(x_)
            freqs_cis = self.broadcast(self.freqs_cis[:x.shape[1]], x_)
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
        return x_out.type_as(x)


class IrregularRoPE(nn.Module):
    """Irregular Rotary Positional Embedding for time-series with irregular timestamps"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.thetas = self.precompute_thetas(d_model)
    
    def precompute_thetas(self, d_model: int, theta: float = 1000.0):
        """Precompute theta values for frequency computation"""
        thetas = 1.0 / (theta ** (torch.arange(0, d_model, 2)[:(d_model//2)].float() / d_model))
        return thetas
    
    def broadcast(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Broadcast frequency components based on timestamps"""
        assert self.thetas.shape[0] == (x.shape[-1])
        shape = [d if i != 2 else 1 for i, d in enumerate(x.shape)]
        freqs_cis = self.compute_freqs_cis(t, shape)
        return freqs_cis.to(x.device)
    
    def compute_freqs_cis(self, t: torch.Tensor, shape: list) -> torch.Tensor:
        """Compute frequency components for given timestamps"""
        f_shape = list(t.shape)
        f_shape.append(self.thetas.shape[0])
        freqs = torch.outer(t.flatten(), self.thetas.to(t.device)).float().reshape(*f_shape)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis.view(*shape)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply irregular RoPE to input tensor with timestamps"""
        with autocast(enabled=False):
            x_ = x.reshape(*x.shape[:-1], -1, 2).float()
            x_ = torch.view_as_complex(x_)
            freqs_cis = self.broadcast(x_, t)
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
        return x_out.type_as(x)