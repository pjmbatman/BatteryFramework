"""
Configuration loader utility that converts YAML configs to proper Config objects
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from .config import Config


def load_config_from_yaml(config_path: str) -> Config:
    """
    Load configuration from YAML file and convert to Config object
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with flattened parameters
    """
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Flatten nested YAML structure to match Config class
    flattened_config = {}
    
    # Model configuration
    if 'model' in yaml_config:
        model_config = yaml_config['model']
        flattened_config['model_name'] = model_config.get('name', 'lipm')
        flattened_config['model_setting'] = model_config.get('setting', 2)
    
    # Data configuration
    if 'data' in yaml_config:
        data_config = yaml_config['data']
        flattened_config['dataset_name'] = data_config.get('dataset_name', 'NASA')
        flattened_config['data_path'] = data_config.get('data_path', 'data/processed')
        flattened_config['datasets'] = data_config.get('datasets', ['NASA'])  # Add datasets field
        flattened_config['patch_len'] = data_config.get('patch_len', 64)
        flattened_config['patch_num'] = data_config.get('patch_num', 16)
        flattened_config['patch_stride'] = data_config.get('patch_stride', -1)
        flattened_config['n_var'] = data_config.get('n_var', 2)
        flattened_config['normalize'] = data_config.get('normalize', True)
    
    # Training configuration
    if 'training' in yaml_config:
        training_config = yaml_config['training']
        flattened_config['batch_size'] = training_config.get('batch_size', 256)
        flattened_config['lr'] = training_config.get('lr', 1e-4)
        flattened_config['l2'] = training_config.get('l2', 1e-3)
        flattened_config['max_epoch'] = training_config.get('max_epoch', 100)
        flattened_config['max_iter'] = training_config.get('max_iter', 50000)
        flattened_config['T_0'] = training_config.get('T_0', 10)
        flattened_config['optimizer'] = training_config.get('optimizer', 'adamw')
        flattened_config['scheduler'] = training_config.get('scheduler', 'cosine_annealing')
    
    # Architecture configuration
    if 'architecture' in yaml_config:
        arch_config = yaml_config['architecture']
        flattened_config['d_model'] = arch_config.get('d_model', 256)
        flattened_config['n_head'] = arch_config.get('n_head', 8)
        flattened_config['n_layer'] = arch_config.get('n_layer', 6)
        flattened_config['emb_dim'] = arch_config.get('emb_dim', 256)
        flattened_config['down_dim'] = arch_config.get('down_dim', 256)
        flattened_config['down_n_head'] = arch_config.get('down_n_head', 4)
        flattened_config['dp'] = arch_config.get('dp', 0.3)
        flattened_config['norm'] = arch_config.get('norm', 'rsm')
        flattened_config['pre_norm'] = arch_config.get('pre_norm', 1)
    
    # Masking configuration
    if 'masking' in yaml_config:
        masking_config = yaml_config['masking']
        flattened_config['channel_ratio'] = masking_config.get('channel_ratio', 0.3)
        flattened_config['patch_ratio'] = masking_config.get('patch_ratio', 0.3)
        flattened_config['weight_MAE'] = masking_config.get('weight_MAE', 1.0)
        flattened_config['weight_Q'] = masking_config.get('weight_Q', 1.0)
    
    # Task configuration
    if 'task' in yaml_config:
        task_config = yaml_config['task']
        flattened_config['task'] = task_config.get('type', 'ir_pretrain')
        flattened_config['downstream_task'] = task_config.get('downstream_task', None)
    
    # Paths configuration
    if 'paths' in yaml_config:
        paths_config = yaml_config['paths']
        flattened_config['checkpoint_dir'] = paths_config.get('checkpoint_dir', 'checkpoints')
        flattened_config['log_dir'] = paths_config.get('log_dir', 'logs')
        flattened_config['output_dir'] = paths_config.get('output_dir', 'outputs')
    
    # Hardware configuration
    if 'hardware' in yaml_config:
        hardware_config = yaml_config['hardware']
        flattened_config['device'] = hardware_config.get('device', 'cuda')
        flattened_config['seed'] = hardware_config.get('seed', 42)
    
    # Create Config object
    config = Config(**flattened_config)
    
    return config