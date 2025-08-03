import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BatteryDataLoader:
    """Battery-specific data loader with utilities for batching and preprocessing"""
    
    def __init__(self, 
                 dataset,
                 config: Config,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        Args:
            dataset: Battery dataset instance
            config: Configuration object
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.config = config
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"DataLoader created with batch_size={config.batch_size}, "
                   f"num_workers={num_workers}")
    
    def _collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        """Custom collate function for battery data batching"""
        vc_batch = []
        qdqc_batch = []
        t_batch = []
        tm_batch = []
        pm_batch = []
        
        for vc, qdqc, t, tm, pm in batch:
            vc_batch.append(vc)
            qdqc_batch.append(qdqc)
            t_batch.append(t)
            tm_batch.append(tm)
            pm_batch.append(pm)
        
        # Stack tensors
        vc_batch = torch.stack(vc_batch)
        qdqc_batch = torch.stack(qdqc_batch)
        t_batch = torch.stack(t_batch)
        tm_batch = torch.stack(tm_batch)
        pm_batch = torch.stack(pm_batch)
        
        return vc_batch, qdqc_batch, t_batch, tm_batch, pm_batch
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    @staticmethod
    def move_batch_to_device(batch: Tuple[torch.Tensor, ...], 
                           device: str) -> Tuple[torch.Tensor, ...]:
        """Move batch tensors to specified device"""
        return tuple(tensor.to(device) for tensor in batch)