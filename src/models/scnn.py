"""
Stacked Convolutional Neural Network (SCNN) for Local State Feature Compression in GraphMARL.

Processes sliding window time-series history of local department resource metrics
(CPU, GPU, RAM, Queue, Power, HSI) into a compact, fixed-size feature vector.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedCNN(nn.Module):
    """
    Stacked 1D-CNN feature extractor for local edge node state time-series.
    """

    def __init__(
        self,
        in_channels: int = 6,
        seq_len: int = 10,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(128, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, In_Channels) or (Batch, In_Channels, Seq_Len)
        Returns:
            Embedding tensor of shape (Batch, Embedding_Dim)
        """
        # Ensure shape is (Batch, In_Channels, Seq_Len)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim if unbatched
        if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
            x = x.permute(0, 2, 1)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out).squeeze(-1)
        out = self.dropout(out)
        out = self.layer_norm(F.relu(self.fc(out)))
        return out
