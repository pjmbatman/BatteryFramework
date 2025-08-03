import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BatteryDataPreprocessor:
    """Preprocessor for battery data with various normalization and filtering options"""
    
    def __init__(self, 
                 normalize_method: str = "zscore",
                 filter_incomplete_cycles: bool = True,
                 min_cycle_length: int = 100,
                 capacity_threshold: float = 0.8):
        """
        Args:
            normalize_method: Normalization method ('zscore', 'minmax', 'none')
            filter_incomplete_cycles: Whether to filter out incomplete charge/discharge cycles
            min_cycle_length: Minimum number of data points per cycle
            capacity_threshold: Threshold for capacity retention filtering
        """
        self.normalize_method = normalize_method
        self.filter_incomplete_cycles = filter_incomplete_cycles
        self.min_cycle_length = min_cycle_length
        self.capacity_threshold = capacity_threshold
        
        self.normalization_stats = {}
    
    def preprocess_cycle_data(self, cycle_data: List[Dict]) -> List[Dict]:
        """Preprocess a list of cycle data"""
        processed_cycles = []
        
        for cycle in cycle_data:
            if self._should_filter_cycle(cycle):
                continue
                
            processed_cycle = self._preprocess_single_cycle(cycle)
            if processed_cycle is not None:
                processed_cycles.append(processed_cycle)
        
        logger.info(f"Preprocessed {len(processed_cycles)} cycles from {len(cycle_data)} original cycles")
        return processed_cycles
    
    def _should_filter_cycle(self, cycle: Dict) -> bool:
        """Determine if a cycle should be filtered out"""
        if self.filter_incomplete_cycles:
            # Check if cycle has both charge and discharge phases
            current = cycle.get('current_in_A', [])
            if not current:
                return True
            
            current_array = np.array(current)
            has_charge = np.any(current_array > 0)
            has_discharge = np.any(current_array < 0)
            
            if not (has_charge and has_discharge):
                return True
        
        # Check minimum cycle length
        voltage = cycle.get('voltage_in_V', [])
        if len(voltage) < self.min_cycle_length:
            return True
        
        return False
    
    def _preprocess_single_cycle(self, cycle: Dict) -> Optional[Dict]:
        """Preprocess a single cycle"""
        try:
            # Extract and validate data
            voltage = np.array(cycle['voltage_in_V'])
            current = np.array(cycle['current_in_A'])
            time = np.array(cycle['time_in_s'])
            
            if len(voltage) != len(current) or len(voltage) != len(time):
                logger.warning("Mismatched array lengths in cycle data")
                return None
            
            # Remove invalid values
            valid_mask = (
                np.isfinite(voltage) & 
                np.isfinite(current) & 
                np.isfinite(time) &
                (voltage > 0)  # Voltage should be positive
            )
            
            if np.sum(valid_mask) < self.min_cycle_length:
                return None
            
            voltage = voltage[valid_mask]
            current = current[valid_mask]
            time = time[valid_mask]
            
            # Normalize time to start from 0
            time = time - time[0]
            
            # Apply normalization
            if self.normalize_method == "zscore":
                voltage = self._zscore_normalize(voltage, "voltage")
                current = self._zscore_normalize(current, "current")
            elif self.normalize_method == "minmax":
                voltage = self._minmax_normalize(voltage, "voltage")
                current = self._minmax_normalize(current, "current")
            
            # Create processed cycle dictionary
            processed_cycle = cycle.copy()
            processed_cycle['voltage_in_V'] = voltage.tolist()
            processed_cycle['current_in_A'] = current.tolist()
            processed_cycle['time_in_s'] = time.tolist()
            
            return processed_cycle
            
        except Exception as e:
            logger.error(f"Error preprocessing cycle: {e}")
            return None
    
    def _zscore_normalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Apply Z-score normalization"""
        if data_type not in self.normalization_stats:
            self.normalization_stats[data_type] = {
                'mean': np.mean(data),
                'std': np.std(data)
            }
        
        stats = self.normalization_stats[data_type]
        return (data - stats['mean']) / (stats['std'] + 1e-8)
    
    def _minmax_normalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Apply Min-Max normalization"""
        if data_type not in self.normalization_stats:
            self.normalization_stats[data_type] = {
                'min': np.min(data),
                'max': np.max(data)
            }
        
        stats = self.normalization_stats[data_type]
        data_range = stats['max'] - stats['min']
        if data_range == 0:
            return data
        return (data - stats['min']) / data_range
    
    def fit_normalization_stats(self, all_cycle_data: List[List[Dict]]):
        """Fit normalization statistics on all data"""
        all_voltages = []
        all_currents = []
        
        for battery_cycles in all_cycle_data:
            for cycle in battery_cycles:
                if not self._should_filter_cycle(cycle):
                    voltage = np.array(cycle['voltage_in_V'])
                    current = np.array(cycle['current_in_A'])
                    
                    # Filter valid values
                    valid_mask = np.isfinite(voltage) & np.isfinite(current) & (voltage > 0)
                    all_voltages.extend(voltage[valid_mask])
                    all_currents.extend(current[valid_mask])
        
        if self.normalize_method == "zscore":
            self.normalization_stats['voltage'] = {
                'mean': np.mean(all_voltages),
                'std': np.std(all_voltages)
            }
            self.normalization_stats['current'] = {
                'mean': np.mean(all_currents),
                'std': np.std(all_currents)
            }
        elif self.normalize_method == "minmax":
            self.normalization_stats['voltage'] = {
                'min': np.min(all_voltages),
                'max': np.max(all_voltages)
            }
            self.normalization_stats['current'] = {
                'min': np.min(all_currents),
                'max': np.max(all_currents)
            }
        
        logger.info(f"Fitted normalization stats using {self.normalize_method} method")
    
    def denormalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Denormalize data back to original scale"""
        if data_type not in self.normalization_stats:
            return data
        
        stats = self.normalization_stats[data_type]
        
        if self.normalize_method == "zscore":
            return data * stats['std'] + stats['mean']
        elif self.normalize_method == "minmax":
            return data * (stats['max'] - stats['min']) + stats['min']
        else:
            return data