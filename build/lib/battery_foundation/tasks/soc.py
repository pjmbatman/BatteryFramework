import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base import BaseDownstreamTask
from ..utils.registry import TaskRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


@TaskRegistry.register("soc")
class SOCPredictor(BaseDownstreamTask):
    """State of Charge (SOC) prediction downstream task"""
    
    def _build_task_head(self):
        """Build SOC prediction head"""
        # Get dimensions from config
        hidden_dim = getattr(self.config, 'soc_hidden_dim', 256)
        num_layers = getattr(self.config, 'soc_num_layers', 3)
        dropout = getattr(self.config, 'soc_dropout', 0.2)
        
        # SOC prediction can be sequence-to-sequence or sequence-to-one
        self.prediction_mode = getattr(self.config, 'soc_prediction_mode', 'sequence')  # 'sequence' or 'point'
        
        # Use LSTM for temporal modeling
        self.use_lstm = getattr(self.config, 'soc_use_lstm', True)
        if self.use_lstm:
            self.lstm_hidden_dim = getattr(self.config, 'soc_lstm_hidden', 128)
            self.lstm_layers = getattr(self.config, 'soc_lstm_layers', 2)
            
            self.lstm = nn.LSTM(
                input_size=self.emb_dim,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_layers,
                batch_first=True,
                dropout=dropout if self.lstm_layers > 1 else 0,
                bidirectional=False
            )
            lstm_output_dim = self.lstm_hidden_dim
        else:
            lstm_output_dim = self.emb_dim
        
        # MLP head for regression
        layers = []
        input_dim = lstm_output_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final prediction layer
        if self.prediction_mode == 'sequence':
            # Predict SOC for each patch
            layers.append(nn.Linear(hidden_dim, 1))
        else:
            # Predict single SOC value
            layers.append(nn.Linear(hidden_dim, 1))
        
        layers.append(nn.Sigmoid())  # SOC is in [0, 1] range
        
        self.prediction_head = nn.Sequential(*layers)
        
        # Loss function with temporal consistency
        self.criterion = nn.MSELoss()
        self.temporal_consistency_weight = getattr(self.config, 'soc_temporal_weight', 0.1)
        
        logger.info(f"SOC predictor: {num_layers} layers, {hidden_dim} hidden dim, "
                   f"mode={self.prediction_mode}, LSTM={self.use_lstm}")
    
    def forward(self, batch) -> torch.Tensor:
        """
        Forward pass for SOC prediction
        
        Args:
            batch: Input batch (vc, qdqc, t, tm, pm)
            
        Returns:
            SOC predictions [batch_size, seq_len, 1] or [batch_size, 1]
        """
        # Get embeddings from backbone
        embeddings = self.get_embeddings(batch)  # [batch, patches, emb_dim]
        
        # Apply LSTM if enabled
        if self.use_lstm:
            lstm_out, _ = self.lstm(embeddings)  # [batch, patches, lstm_hidden_dim]
            features = lstm_out
        else:
            features = embeddings
        
        # Predict SOC
        if self.prediction_mode == 'sequence':
            # Sequence-to-sequence prediction
            soc_pred = self.prediction_head(features)  # [batch, patches, 1]
        else:
            # Use last timestep for point prediction
            last_features = features[:, -1, :]  # [batch, lstm_hidden_dim]
            soc_pred = self.prediction_head(last_features)  # [batch, 1]
        
        return soc_pred
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    batch: Tuple = None) -> torch.Tensor:
        """
        Compute SOC prediction loss with temporal consistency
        
        Args:
            predictions: Predicted SOC values
            targets: Target SOC values
            batch: Original batch for temporal information
            
        Returns:
            Combined loss (MSE + temporal consistency)
        """
        # Basic MSE loss
        mse_loss = self.criterion(predictions.squeeze(), targets.squeeze())
        
        # Temporal consistency loss (smooth SOC changes)
        if self.prediction_mode == 'sequence' and predictions.shape[1] > 1:
            # Compute differences between consecutive predictions
            pred_diff = torch.diff(predictions, dim=1)
            target_diff = torch.diff(targets, dim=1)
            
            temporal_loss = self.criterion(pred_diff.squeeze(), target_diff.squeeze())
            
            total_loss = mse_loss + self.temporal_consistency_weight * temporal_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def predict_soc_trajectory(self, batch) -> Dict[str, torch.Tensor]:
        """
        Predict SOC trajectory with additional analysis
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary with SOC predictions and analysis
        """
        self.eval()
        with torch.no_grad():
            # Get base prediction
            soc_pred = self.forward(batch)
            
            # Get embeddings for analysis
            embeddings = self.get_embeddings(batch)
            
            if self.use_lstm:
                lstm_out, (hidden, cell) = self.lstm(embeddings)
                features = lstm_out
            else:
                features = embeddings
            
            # Compute prediction confidence (based on feature variance)
            feature_var = torch.var(features, dim=-1)  # [batch, patches]
            confidence = 1.0 / (feature_var + 1e-6)
            
            # Estimate charge/discharge phases
            if self.prediction_mode == 'sequence' and soc_pred.shape[1] > 1:
                soc_diff = torch.diff(soc_pred, dim=1)
                charge_phase = (soc_diff > 0.01).float()  # Threshold for charging
                discharge_phase = (soc_diff < -0.01).float()  # Threshold for discharging
            else:
                charge_phase = torch.zeros_like(soc_pred)
                discharge_phase = torch.zeros_like(soc_pred)
            
            return {
                'soc_prediction': soc_pred,
                'confidence': confidence,
                'charge_phase': charge_phase,
                'discharge_phase': discharge_phase,
                'soc_change_rate': soc_diff if self.prediction_mode == 'sequence' and soc_pred.shape[1] > 1 else None
            }
    
    def estimate_remaining_capacity(self, batch, current_soc: float = None) -> Dict[str, torch.Tensor]:
        """
        Estimate remaining usable capacity based on SOC prediction
        
        Args:
            batch: Input batch
            current_soc: Current SOC value (if known)
            
        Returns:
            Dictionary with capacity estimates
        """
        self.eval()
        with torch.no_grad():
            # Get SOC trajectory
            results = self.predict_soc_trajectory(batch)
            soc_pred = results['soc_prediction']
            
            if current_soc is not None:
                # Use provided current SOC
                current_soc_tensor = torch.full_like(soc_pred[:, 0:1], current_soc)
            else:
                # Use first prediction as current SOC
                current_soc_tensor = soc_pred[:, 0:1] if soc_pred.dim() > 1 else soc_pred
            
            # Estimate usable capacity (from current SOC to minimum usable SOC)
            min_usable_soc = getattr(self.config, 'min_usable_soc', 0.1)
            remaining_capacity_ratio = torch.clamp(current_soc_tensor - min_usable_soc, min=0.0)
            
            # Estimate time to discharge (if in discharge phase)
            if self.prediction_mode == 'sequence' and soc_pred.shape[1] > 1:
                discharge_rate = torch.mean(results['soc_change_rate'][results['discharge_phase'] > 0.5])
                time_to_empty = remaining_capacity_ratio / torch.abs(discharge_rate) if discharge_rate < 0 else torch.inf
            else:
                time_to_empty = torch.full_like(remaining_capacity_ratio, float('inf'))
            
            return {
                'remaining_capacity_ratio': remaining_capacity_ratio,
                'time_to_empty_cycles': time_to_empty,
                'current_soc': current_soc_tensor,
                'min_usable_soc': min_usable_soc
            }


@TaskRegistry.register("soc_estimation")
class SOCEstimator(SOCPredictor):
    """Real-time SOC estimation variant with Kalman filter-like updates"""
    
    def _build_task_head(self):
        """Build SOC estimator with state tracking"""
        super()._build_task_head()
        
        # Add state estimation components
        self.state_dim = getattr(self.config, 'soc_state_dim', 32)
        
        # State update mechanism
        self.state_update = nn.Linear(self.emb_dim + self.state_dim, self.state_dim)
        self.state_to_soc = nn.Linear(self.state_dim, 1)
        
        # Initialize state
        self.register_buffer('initial_state', torch.zeros(1, self.state_dim))
        
        logger.info(f"SOC estimator with {self.state_dim}-dimensional state tracking")
    
    def forward_with_state(self, batch, previous_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with state tracking
        
        Args:
            batch: Input batch
            previous_state: Previous internal state
            
        Returns:
            SOC prediction and updated state
        """
        # Get embeddings
        embeddings = self.get_embeddings(batch)  # [batch, patches, emb_dim]
        
        if previous_state is None:
            previous_state = self.initial_state.expand(embeddings.shape[0], -1)
        
        # Process each timestep sequentially
        states = []
        soc_predictions = []
        
        current_state = previous_state
        
        for t in range(embeddings.shape[1]):
            # Combine current embedding with previous state
            combined_input = torch.cat([embeddings[:, t, :], current_state], dim=1)
            
            # Update state
            current_state = torch.tanh(self.state_update(combined_input))
            states.append(current_state)
            
            # Predict SOC from current state
            soc_pred = torch.sigmoid(self.state_to_soc(current_state))
            soc_predictions.append(soc_pred)
        
        # Stack predictions
        soc_sequence = torch.stack(soc_predictions, dim=1)  # [batch, patches, 1]
        final_state = current_state
        
        return soc_sequence, final_state