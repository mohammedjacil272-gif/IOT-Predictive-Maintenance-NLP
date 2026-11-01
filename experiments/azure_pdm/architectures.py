"""Azure-specific alternative compact classifier architectures."""

from __future__ import annotations

import torch
from torch import nn


class _Base(nn.Module):
    def __init__(self, metadata_dim: int, class_count: int = 6) -> None:
        super().__init__()
        self.metadata_dim = metadata_dim
        self.class_count = class_count
        self.metadata_encoder = nn.Sequential(nn.Linear(metadata_dim, 12), nn.ReLU())

    def _metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        if metadata.ndim != 2 or metadata.shape[1] != self.metadata_dim:
            raise ValueError("metadata shape mismatch")
        return self.metadata_encoder(metadata)


class AzureLSTMClassifier(_Base):
    def __init__(self, metadata_dim: int, class_count: int = 6) -> None:
        super().__init__(metadata_dim, class_count)
        self.encoder = nn.LSTM(4, 20, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Dropout(0.15), nn.Linear(24, class_count))

    def forward(self, telemetry: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if telemetry.ndim != 3 or telemetry.shape[2] != 4:
            raise ValueError("telemetry shape mismatch")
        if not torch.isfinite(telemetry).all() or not torch.isfinite(metadata).all():
            raise ValueError("inputs must be finite")
        _, (hidden, _) = self.encoder(telemetry)
        return self.head(torch.cat([hidden[-1], self._metadata(metadata)], dim=1))


class AzureGRUClassifier(_Base):
    def __init__(self, metadata_dim: int, class_count: int = 6) -> None:
        super().__init__(metadata_dim, class_count)
        self.encoder = nn.GRU(4, 20, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Dropout(0.15), nn.Linear(24, class_count))

    def forward(self, telemetry: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if telemetry.ndim != 3 or telemetry.shape[2] != 4:
            raise ValueError("telemetry shape mismatch")
        if not torch.isfinite(telemetry).all() or not torch.isfinite(metadata).all():
            raise ValueError("inputs must be finite")
        _, hidden = self.encoder(telemetry)
        return self.head(torch.cat([hidden[-1], self._metadata(metadata)], dim=1))


class AzureAggregateMLPClassifier(_Base):
    def __init__(self, metadata_dim: int, class_count: int = 6) -> None:
        super().__init__(metadata_dim, class_count)
        self.head = nn.Sequential(nn.Linear(8 + 12, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, class_count))

    def forward(self, telemetry: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if telemetry.ndim != 3 or telemetry.shape[2] != 4:
            raise ValueError("telemetry shape mismatch")
        if not torch.isfinite(telemetry).all() or not torch.isfinite(metadata).all():
            raise ValueError("inputs must be finite")
        aggregate = torch.cat([telemetry.mean(dim=1), telemetry.std(dim=1, unbiased=False)], dim=1)
        return self.head(torch.cat([aggregate, self._metadata(metadata)], dim=1))
