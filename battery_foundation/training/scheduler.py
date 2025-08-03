import torch
from typing import Optional

from ..utils.config import Config
from ..utils.registry import SchedulerRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


@SchedulerRegistry.register("cosine_annealing")
def get_cosine_annealing_scheduler(optimizer: torch.optim.Optimizer, 
                                  config: Config) -> torch.optim.lr_scheduler._LRScheduler:
    """Get cosine annealing with warm restarts scheduler"""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.T_0
    )


@SchedulerRegistry.register("step")
def get_step_scheduler(optimizer: torch.optim.Optimizer, 
                      config: Config) -> torch.optim.lr_scheduler._LRScheduler:
    """Get step learning rate scheduler"""
    step_size = getattr(config, 'step_size', 1000)
    gamma = getattr(config, 'gamma', 0.5)
    
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )


@SchedulerRegistry.register("exponential")
def get_exponential_scheduler(optimizer: torch.optim.Optimizer, 
                             config: Config) -> torch.optim.lr_scheduler._LRScheduler:
    """Get exponential learning rate scheduler"""
    gamma = getattr(config, 'gamma', 0.95)
    
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma
    )


@SchedulerRegistry.register("reduce_on_plateau")
def get_reduce_on_plateau_scheduler(optimizer: torch.optim.Optimizer, 
                                   config: Config) -> torch.optim.lr_scheduler._LRScheduler:
    """Get reduce on plateau scheduler"""
    patience = getattr(config, 'patience', 10)
    factor = getattr(config, 'factor', 0.5)
    
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        verbose=True
    )


def get_scheduler(optimizer: torch.optim.Optimizer, 
                 config: Config) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler based on configuration
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration object
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = getattr(config, 'scheduler', 'cosine_annealing').lower()
    
    if scheduler_name == 'none' or scheduler_name is None:
        logger.info("No learning rate scheduler will be used")
        return None
    
    if scheduler_name not in SchedulerRegistry:
        logger.warning(f"Scheduler '{scheduler_name}' not found, using 'cosine_annealing'")
        scheduler_name = 'cosine_annealing'
    
    scheduler_fn = SchedulerRegistry.get(scheduler_name)
    scheduler = scheduler_fn(optimizer, config)
    
    logger.info(f"Using scheduler: {scheduler.__class__.__name__}")
    
    return scheduler