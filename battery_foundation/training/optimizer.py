import torch
from typing import Dict, Any

from ..utils.config import Config
from ..utils.registry import OptimizerRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


@OptimizerRegistry.register("adamw")
def get_adamw_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    """Get AdamW optimizer"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.l2,
        betas=[0.9, 0.95]
    )


@OptimizerRegistry.register("adam")
def get_adam_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    """Get Adam optimizer"""
    return torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.l2
    )


@OptimizerRegistry.register("sgd")
def get_sgd_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    """Get SGD optimizer"""
    return torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.l2,
        momentum=0.9
    )


def get_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    """
    Get optimizer based on configuration
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        Optimizer instance
    """
    optimizer_name = getattr(config, 'optimizer', 'adamw').lower()
    
    if optimizer_name not in OptimizerRegistry:
        logger.warning(f"Optimizer '{optimizer_name}' not found, using 'adamw'")
        optimizer_name = 'adamw'
    
    optimizer_fn = OptimizerRegistry.get(optimizer_name)
    optimizer = optimizer_fn(model, config)
    
    logger.info(f"Using optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Weight decay: {config.l2}")
    
    return optimizer