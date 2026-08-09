"""
Predictive Congestion-Aware Scheduling (PCAS) module for GraphMARL.

Provides ultra-lightweight auto-regressive queue length and bottleneck forecasting
to anticipate edge department congestion before buffer saturation occurs.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PCASCongestionForecaster(nn.Module):
    """
    Lightweight Auto-Regressive neural forecaster for edge queue congestion.
    """

    def __init__(
        self,
        history_window: int = 10,
        forecast_horizon: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.history_window = history_window
        self.forecast_horizon = forecast_horizon

        # Input: history of queue lengths (history_window) + delta + arrival_rate
        in_dim = history_window + 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, forecast_horizon + 1),  # Predictions + Congestion Probability
        )

    def forward(self, queue_history: torch.Tensor, arrival_rate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queue_history: (Batch, history_window)
            arrival_rate: (Batch, 1) or scalar
        Returns:
            predicted_queues: (Batch, forecast_horizon)
            congestion_prob: (Batch, 1) in [0.0, 1.0]
        """
        if queue_history.dim() == 1:
            queue_history = queue_history.unsqueeze(0)
        if arrival_rate.dim() == 0:
            arrival_rate = arrival_rate.view(1, 1)
        elif arrival_rate.dim() == 1:
            arrival_rate = arrival_rate.unsqueeze(-1)

        batch_size = queue_history.shape[0]

        # Compute delta between last two history points
        if queue_history.shape[1] >= 2:
            delta = (queue_history[:, -1:] - queue_history[:, -2:-1])
        else:
            delta = torch.zeros(batch_size, 1, device=queue_history.device)

        inputs = torch.cat([queue_history, delta, arrival_rate], dim=-1)
        out = self.net(inputs)

        pred_queues = F.relu(out[:, : self.forecast_horizon])
        congestion_prob = torch.sigmoid(out[:, self.forecast_horizon :])

        return pred_queues, congestion_prob

    def predict_numpy(self, queue_history_list: list, arrival_rate: float = 1.0) -> Tuple[np.ndarray, float]:
        """Numpy convenience method for environment evaluation."""
        hist = np.array(queue_history_list[-self.history_window :], dtype=np.float32)
        if len(hist) < self.history_window:
            hist = np.pad(hist, (self.history_window - len(hist), 0), mode="constant")
        with torch.no_grad():
            t_hist = torch.from_numpy(hist).unsqueeze(0)
            t_arr = torch.tensor([[arrival_rate]], dtype=torch.float32)
            preds, prob = self.forward(t_hist, t_arr)
            return preds.squeeze(0).numpy(), float(prob.squeeze().item())
