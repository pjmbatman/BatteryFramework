from typing import List, Dict, Any
from collections import defaultdict
import numpy as np


class LossTracker:
    """Track and compute statistics for training losses"""
    
    def __init__(self, loss_keys: List[str]):
        """
        Initialize loss tracker
        
        Args:
            loss_keys: List of loss keys to track
        """
        self.loss_keys = loss_keys
        self.losses = defaultdict(list)
        self.reset()
    
    def update(self, loss_dict: Dict[str, float]):
        """Update losses with new values"""
        for key in self.loss_keys:
            if key in loss_dict:
                self.losses[key].append(loss_dict[key])
    
    def get_average(self) -> Dict[str, float]:
        """Get average of all tracked losses"""
        avg_losses = {}
        for key in self.loss_keys:
            if self.losses[key]:
                avg_losses[key] = np.mean(self.losses[key])
            else:
                avg_losses[key] = 0.0
        return avg_losses
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values of all tracked losses"""
        latest_losses = {}
        for key in self.loss_keys:
            if self.losses[key]:
                latest_losses[key] = self.losses[key][-1]
            else:
                latest_losses[key] = 0.0
        return latest_losses
    
    def get_std(self) -> Dict[str, float]:
        """Get standard deviation of all tracked losses"""
        std_losses = {}
        for key in self.loss_keys:
            if len(self.losses[key]) > 1:
                std_losses[key] = np.std(self.losses[key])
            else:
                std_losses[key] = 0.0
        return std_losses
    
    def get_min(self) -> Dict[str, float]:
        """Get minimum values of all tracked losses"""
        min_losses = {}
        for key in self.loss_keys:
            if self.losses[key]:
                min_losses[key] = np.min(self.losses[key])
            else:
                min_losses[key] = float('inf')
        return min_losses
    
    def get_max(self) -> Dict[str, float]:
        """Get maximum values of all tracked losses"""
        max_losses = {}
        for key in self.loss_keys:
            if self.losses[key]:
                max_losses[key] = np.max(self.losses[key])
            else:
                max_losses[key] = 0.0
        return max_losses
    
    def get_count(self) -> int:
        """Get number of updates"""
        if self.loss_keys and self.losses[self.loss_keys[0]]:
            return len(self.losses[self.loss_keys[0]])
        return 0
    
    def reset(self):
        """Reset all tracked losses"""
        for key in self.loss_keys:
            self.losses[key] = []
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary of all losses"""
        summary = {}
        for key in self.loss_keys:
            summary[key] = {
                'mean': np.mean(self.losses[key]) if self.losses[key] else 0.0,
                'std': np.std(self.losses[key]) if len(self.losses[key]) > 1 else 0.0,
                'min': np.min(self.losses[key]) if self.losses[key] else float('inf'),
                'max': np.max(self.losses[key]) if self.losses[key] else 0.0,
                'count': len(self.losses[key])
            }
        return summary