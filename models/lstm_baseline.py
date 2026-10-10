"""PyTorch LSTM baseline for FD001 RUL regression."""

from __future__ import annotations

import torch
from torch import nn


class LSTMBaseline(nn.Module):
    """Sensor-only LSTM regressor with a 64-dimensional embedding."""

    def __init__(
        self,
        input_size: int = 17,
        hidden_size: int = 64,
        num_layers: int = 2,
        lstm_dropout: float = 0.2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(32, 1),
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_embedding(x)
        return self.regression_head(embedding).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
