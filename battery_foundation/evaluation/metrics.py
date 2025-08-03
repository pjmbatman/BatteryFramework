import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BatteryMetrics:
    """Comprehensive metrics for battery model evaluation"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared Score"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Maximum absolute error"""
        return np.max(np.abs(y_true - y_pred))
    
    @staticmethod
    def capacity_accuracy_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, 
                                     threshold: float = 0.8) -> Dict[str, float]:
        """
        Accuracy metrics for capacity prediction at specific thresholds
        
        Args:
            y_true: True capacity values (normalized 0-1)
            y_pred: Predicted capacity values
            threshold: Capacity threshold for EOL prediction
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Binary classification at threshold
        true_eol = y_true <= threshold
        pred_eol = y_pred <= threshold
        
        # True/False positives/negatives
        tp = np.sum((true_eol == True) & (pred_eol == True))
        tn = np.sum((true_eol == False) & (pred_eol == False))
        fp = np.sum((true_eol == False) & (pred_eol == True))
        fn = np.sum((true_eol == True) & (pred_eol == False))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    @staticmethod
    def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Remaining Useful Life (RUL) specific metrics
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            
        Returns:
            Dictionary with RUL-specific metrics
        """
        errors = y_pred - y_true
        
        # Early prediction penalty (predicting failure too early)
        early_errors = errors[errors < 0]
        early_penalty = np.sum(np.exp(-early_errors / 10)) if len(early_errors) > 0 else 0
        
        # Late prediction penalty (predicting failure too late)
        late_errors = errors[errors >= 0]
        late_penalty = np.sum(np.exp(late_errors / 13)) if len(late_errors) > 0 else 0
        
        # Prognostic horizon (how far in advance can we predict)
        alpha = 0.2  # Acceptable error fraction
        valid_predictions = np.abs(errors) <= alpha * y_true
        prognostic_horizon = np.mean(y_true[valid_predictions]) if np.any(valid_predictions) else 0
        
        return {
            'early_penalty': early_penalty,
            'late_penalty': late_penalty,
            'total_penalty': early_penalty + late_penalty,
            'prognostic_horizon': prognostic_horizon,
            'valid_prediction_ratio': np.mean(valid_predictions)
        }
    
    @staticmethod
    def soh_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        State of Health (SOH) specific metrics
        
        Args:
            y_true: True SOH values (0-1 scale)
            y_pred: Predicted SOH values
            
        Returns:
            Dictionary with SOH-specific metrics
        """
        # Standard regression metrics
        metrics = {
            'mse': BatteryMetrics.mse(y_true, y_pred),
            'mae': BatteryMetrics.mae(y_true, y_pred),
            'rmse': BatteryMetrics.rmse(y_true, y_pred),
            'mape': BatteryMetrics.mape(y_true, y_pred),
            'r2': BatteryMetrics.r2(y_true, y_pred),
            'max_error': BatteryMetrics.max_error(y_true, y_pred)
        }
        
        # SOH-specific thresholds
        for threshold in [0.9, 0.8, 0.7]:
            thresh_metrics = BatteryMetrics.capacity_accuracy_at_threshold(
                y_true, y_pred, threshold
            )
            for key, value in thresh_metrics.items():
                metrics[f'{key}_at_{threshold}'] = value
        
        return metrics
    
    @staticmethod
    def soc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        State of Charge (SOC) specific metrics
        
        Args:
            y_true: True SOC values (0-1 scale)
            y_pred: Predicted SOC values
            
        Returns:
            Dictionary with SOC-specific metrics
        """
        # Standard regression metrics
        metrics = {
            'mse': BatteryMetrics.mse(y_true, y_pred),
            'mae': BatteryMetrics.mae(y_true, y_pred),
            'rmse': BatteryMetrics.rmse(y_true, y_pred),
            'mape': BatteryMetrics.mape(y_true, y_pred),
            'r2': BatteryMetrics.r2(y_true, y_pred),
            'max_error': BatteryMetrics.max_error(y_true, y_pred)
        }
        
        # SOC-specific analysis
        # Charge/discharge accuracy
        charge_mask = np.diff(y_true, prepend=y_true[0]) >= 0
        discharge_mask = ~charge_mask
        
        if np.any(charge_mask):
            metrics['mae_charging'] = BatteryMetrics.mae(
                y_true[charge_mask], y_pred[charge_mask]
            )
        
        if np.any(discharge_mask):
            metrics['mae_discharging'] = BatteryMetrics.mae(
                y_true[discharge_mask], y_pred[discharge_mask]
            )
        
        # Range-specific accuracy
        low_soc = y_true <= 0.3
        mid_soc = (y_true > 0.3) & (y_true <= 0.7)
        high_soc = y_true > 0.7
        
        for range_name, mask in [('low', low_soc), ('mid', mid_soc), ('high', high_soc)]:
            if np.any(mask):
                metrics[f'mae_{range_name}_soc'] = BatteryMetrics.mae(
                    y_true[mask], y_pred[mask]
                )
        
        return metrics
    
    @staticmethod
    def reconstruction_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, 
                             mask: torch.Tensor = None) -> Dict[str, float]:
        """
        Metrics for reconstruction tasks (voltage/current prediction)
        
        Args:
            y_true: True values [batch, seq, vars]
            y_pred: Predicted values [batch, seq, vars]
            mask: Optional mask for valid positions
            
        Returns:
            Dictionary with reconstruction metrics
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        metrics = {}
        
        # Overall metrics
        if mask is not None:
            # Apply mask
            valid_true = y_true[mask > 0]
            valid_pred = y_pred[mask > 0]
        else:
            valid_true = y_true.flatten()
            valid_pred = y_pred.flatten()
        
        metrics['overall_mse'] = BatteryMetrics.mse(valid_true, valid_pred)
        metrics['overall_mae'] = BatteryMetrics.mae(valid_true, valid_pred)
        metrics['overall_rmse'] = BatteryMetrics.rmse(valid_true, valid_pred)
        
        # Variable-specific metrics (assuming voltage and current)
        if y_true.shape[-1] >= 2:
            for i, var_name in enumerate(['voltage', 'current']):
                if mask is not None:
                    var_true = y_true[..., i][mask[..., i] > 0]
                    var_pred = y_pred[..., i][mask[..., i] > 0]
                else:
                    var_true = y_true[..., i].flatten()
                    var_pred = y_pred[..., i].flatten()
                
                if len(var_true) > 0:
                    metrics[f'{var_name}_mse'] = BatteryMetrics.mse(var_true, var_pred)
                    metrics[f'{var_name}_mae'] = BatteryMetrics.mae(var_true, var_pred)
                    metrics[f'{var_name}_rmse'] = BatteryMetrics.rmse(var_true, var_pred)
        
        return metrics
    
    @staticmethod
    def compute_all_metrics(task_type: str, y_true: np.ndarray, 
                          y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Compute all relevant metrics for a given task type
        
        Args:
            task_type: Type of task ('soh', 'soc', 'rul', 'reconstruction')
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dictionary with all relevant metrics
        """
        if task_type.lower() == 'soh':
            return BatteryMetrics.soh_metrics(y_true, y_pred)
        elif task_type.lower() == 'soc':
            return BatteryMetrics.soc_metrics(y_true, y_pred)
        elif task_type.lower() == 'rul':
            return BatteryMetrics.rul_metrics(y_true, y_pred)
        elif task_type.lower() == 'reconstruction':
            mask = kwargs.get('mask', None)
            return BatteryMetrics.reconstruction_metrics(y_true, y_pred, mask)
        else:
            # Default to basic regression metrics
            return {
                'mse': BatteryMetrics.mse(y_true, y_pred),
                'mae': BatteryMetrics.mae(y_true, y_pred),
                'rmse': BatteryMetrics.rmse(y_true, y_pred),
                'r2': BatteryMetrics.r2(y_true, y_pred)
            }