"""LSTM sequence encoder with a linear head for supervised-then-truncate training."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM over a history window; `encode` returns final hidden state h_n."""

    def __init__(
        self,
        d_in: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> z: (B, hidden_size)."""
        out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, B, H) — take the top layer
        return h_n[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> pred: (B,) standardised log-return."""
        z = self.encode(x)
        return self.head(z).squeeze(-1)
