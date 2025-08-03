import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union, List
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for Battery Foundation Framework"""
    
    # Model configuration
    model_name: str = "lipm"
    model_setting: int = 2
    d_model: int = 256
    n_head: int = 8
    n_layer: int = 6
    emb_dim: int = 256
    down_dim: int = 256
    down_n_head: int = 4
    
    # Data configuration
    dataset_name: str = "battery"
    data_path: str = "../data/processed"
    datasets: List[str] = field(default_factory=lambda: ["NASA"])  # Support multiple datasets
    patch_len: int = 64
    patch_num: int = 16
    patch_stride: int = -1
    n_var: int = 2
    normalize: bool = True
    
    # Training configuration
    batch_size: int = 256
    lr: float = 1e-4
    l2: float = 1e-3
    max_epoch: int = 100
    max_iter: int = 50000
    optimizer: str = "adamw"
    scheduler: str = "cosine_annealing"
    T_0: int = 10
    weight_MAE: float = 1.0
    weight_Q: float = 1.0
    
    # Masking configuration
    channel_ratio: float = 0.3
    patch_ratio: float = 0.3
    
    # Task configuration
    task: str = "ir_pretrain"
    downstream_task: str = None
    
    # Other configurations
    dp: float = 0.3
    norm: str = "rsm"
    pre_norm: int = 1
    device: str = "cuda"
    seed: int = 42
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    
    def __post_init__(self):
        """Apply model-specific settings after initialization"""
        if hasattr(self, 'model_setting'):
            self._apply_model_setting(self.model_setting)
    
    def _apply_model_setting(self, setting_idx: int):
        """Apply predefined model settings similar to LiPM"""
        settings = [
            {"d_model": 64, "n_head": 4, "n_layer": 2},
            {"d_model": 128, "n_head": 4, "n_layer": 3},
            {"d_model": 256, "n_head": 8, "n_layer": 6},
            {"d_model": 512, "n_head": 8, "n_layer": 12},
            {"d_model": 768, "n_head": 12, "n_layer": 18},
            {"d_model": 1024, "n_head": 16, "n_layer": 24}
        ]
        
        if 0 <= setting_idx < len(settings):
            setting = settings[setting_idx]
            self.d_model = setting["d_model"]
            self.n_head = setting["n_head"]
            self.n_layer = setting["n_layer"]
            self.emb_dim = self.d_model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        return cls(**config_dict)


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def save_config(config: Config, save_path: Union[str, Path]):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)