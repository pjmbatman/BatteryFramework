import os
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset

from ..utils.registry import DatasetRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BatteryDataset(Dataset):
    """Base Battery Dataset class for loading and processing battery data"""
    
    def __init__(self, 
                 data_path: str,
                 dataset_names: List[str] = None,
                 patch_len: int = 64,
                 patch_num: int = 16,
                 patch_stride: int = -1,
                 n_var: int = 2,
                 normalize: bool = True):
        """
        Args:
            data_path: Path to processed battery data
            dataset_names: List of dataset names to load (e.g., ['NASA', 'MATR'])
            patch_len: Length of each patch
            patch_num: Number of patches per sample
            patch_stride: Stride for patching (-1 for auto)
            n_var: Number of variables (2 for V,I)
            normalize: Whether to normalize the data
        """
        self.data_path = Path(data_path)
        self.dataset_names = dataset_names or ['NASA']
        self.patch_len = patch_len
        self.patch_num = patch_num
        self.patch_stride = patch_stride if patch_stride > 0 else patch_len
        self.n_var = n_var
        self.normalize = normalize
        
        self.data = []
        self.metadata = []
        
        self._load_data()
        self._compute_stats()
        
        logger.info(f"Loaded {len(self.data)} samples from {self.dataset_names}")
    
    def _load_data(self):
        """Load battery data from pickle files"""
        for dataset_name in self.dataset_names:
            dataset_path = self.data_path / dataset_name
            if not dataset_path.exists():
                logger.warning(f"Dataset path not found: {dataset_path}")
                continue
            
            pickle_files = list(dataset_path.glob("*.pkl"))
            logger.info(f"Loading {len(pickle_files)} files from {dataset_name}")
            
            for pickle_file in pickle_files:
                try:
                    with open(pickle_file, 'rb') as f:
                        battery_data = pickle.load(f)
                    
                    processed_data = self._process_battery_data(battery_data)
                    if processed_data:
                        self.data.extend(processed_data)
                        self.metadata.extend([{
                            'cell_id': battery_data.get('cell_id', str(pickle_file.stem)),
                            'dataset': dataset_name,
                            'file_path': str(pickle_file)
                        }] * len(processed_data))
                        
                except Exception as e:
                    logger.error(f"Error loading {pickle_file}: {e}")
    
    def _process_battery_data(self, battery_data: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        """Process raw battery data into patches suitable for LiPM model"""
        if 'cycle_data' not in battery_data:
            return []
        
        cycle_data = battery_data['cycle_data']
        
        # Concatenate all cycles into a continuous time series
        all_voltage = []
        all_current = []
        all_time = []
        all_q_discharge = []
        all_q_charge = []
        
        for cycle in cycle_data:
            try:
                voltage = np.array(cycle['voltage_in_V'])
                current = np.array(cycle['current_in_A'])
                time = np.array(cycle['time_in_s'])
                
                # Get capacity values for this cycle
                charge_capacity = cycle.get('charge_capacity_in_Ah', [0])
                discharge_capacity = cycle.get('discharge_capacity_in_Ah', [0])
                
                if isinstance(charge_capacity, list) and len(charge_capacity) > 0:
                    q_charge = max(charge_capacity) if charge_capacity else 0
                else:
                    q_charge = charge_capacity if isinstance(charge_capacity, (int, float)) else 0
                
                if isinstance(discharge_capacity, list) and len(discharge_capacity) > 0:
                    q_discharge = max(discharge_capacity) if discharge_capacity else 0
                else:
                    q_discharge = discharge_capacity if isinstance(discharge_capacity, (int, float)) else 0
                
                # Append to continuous series
                all_voltage.extend(voltage)
                all_current.extend(current)
                all_time.extend(time)
                # Repeat Q values for each time step in this cycle
                all_q_discharge.extend([q_discharge] * len(voltage))
                all_q_charge.extend([q_charge] * len(voltage))
                
            except Exception as e:
                logger.warning(f"Error processing cycle: {e}")
                continue
        
        if len(all_voltage) < self.patch_len * self.patch_num:
            return []
        
        # Convert to numpy arrays
        all_voltage = np.array(all_voltage)
        all_current = np.array(all_current)
        all_q_discharge = np.array(all_q_discharge)
        all_q_charge = np.array(all_q_charge)
        
        # Combine V and I as the two variables
        vc_data = np.stack([all_voltage, all_current], axis=-1)  # Shape: (time_steps, 2)
        qdqc_data = np.stack([all_q_discharge, all_q_charge], axis=-1)  # Shape: (time_steps, 2)
        
        # Create multiple samples by sliding window
        processed_samples = []
        total_length = len(vc_data)
        sample_length = self.patch_len * self.patch_num
        
        # Create overlapping windows
        step_size = sample_length // 2  # 50% overlap
        for start_idx in range(0, total_length - sample_length + 1, step_size):
            end_idx = start_idx + sample_length
            
            # Extract window
            vc_window = vc_data[start_idx:end_idx]
            qdqc_window = qdqc_data[start_idx:end_idx]
            
            # Create patches from window
            patches = self._create_patches(vc_window)
            if patches is None:
                continue
            
            # Average Q values over patches
            qdqc_patches = []
            for i in range(self.patch_num):
                patch_start = i * self.patch_len
                patch_end = patch_start + self.patch_len
                q_patch = np.mean(qdqc_window[patch_start:patch_end], axis=0)
                qdqc_patches.append(q_patch)
            qdqc_patches = np.array(qdqc_patches)
            
            # Create time and mask tensors
            # t should be [patch_num, patch_len] for time indices within each patch
            t = torch.arange(self.patch_len).float().unsqueeze(0).repeat(self.patch_num, 1)
            # tm should be [patch_num, patch_len] as time mask for each patch  
            tm = torch.ones(self.patch_num, self.patch_len, dtype=torch.bool)
            # pm should be [patch_num] as patch mask
            pm = torch.ones(self.patch_num, dtype=torch.bool)
            
            processed_samples.append({
                'vc': torch.tensor(patches, dtype=torch.float32),
                'qdqc': torch.tensor(qdqc_patches, dtype=torch.float32),
                't': t,
                'tm': tm,
                'pm': pm
            })
        
        return processed_samples
    
    def _create_patches(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Create patches from continuous time series data"""
        seq_len = len(data)
        total_patch_len = self.patch_len * self.patch_num
        
        if seq_len < total_patch_len:
            return None
        
        # Create patches with stride
        patches = []
        for i in range(self.patch_num):
            start_idx = i * self.patch_stride
            end_idx = start_idx + self.patch_len
            
            if end_idx > seq_len:
                # Pad if necessary
                patch = np.zeros((self.patch_len, self.n_var))
                available_len = seq_len - start_idx
                if available_len > 0:
                    patch[:available_len] = data[start_idx:seq_len]
            else:
                patch = data[start_idx:end_idx]
            
            patches.append(patch)
        
        return np.array(patches)  # Shape: (patch_num, patch_len, n_var)
    
    def _compute_stats(self):
        """Compute normalization statistics"""
        if not self.data or not self.normalize:
            self.mean = 0
            self.std = 1
            return
        
        all_vc = torch.cat([item['vc'].reshape(-1, self.n_var) for item in self.data], dim=0)
        self.mean = all_vc.mean(dim=0)
        self.std = all_vc.std(dim=0)
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sample in the format expected by LiPM model"""
        item = self.data[idx]
        
        vc = item['vc']
        qdqc = item['qdqc']
        t = item['t']
        tm = item['tm']
        pm = item['pm']
        
        # Normalize if enabled
        if self.normalize:
            vc = (vc - self.mean) / self.std
        
        return vc, qdqc, t, tm, pm


@DatasetRegistry.register("battery")
class DefaultBatteryDataset(BatteryDataset):
    """Default battery dataset implementation"""
    pass


@DatasetRegistry.register("nasa")
class NASADataset(BatteryDataset):
    """NASA-specific battery dataset"""
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, dataset_names=['NASA'], **kwargs)


@DatasetRegistry.register("matr")
class MATRDataset(BatteryDataset):
    """MATR-specific battery dataset"""
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, dataset_names=['MATR'], **kwargs)