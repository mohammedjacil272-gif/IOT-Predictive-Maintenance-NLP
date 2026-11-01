"""Compact Azure PdM telemetry + metadata classifier."""

from __future__ import annotations

import torch
from torch import nn


class AzureTemporalClassifier(nn.Module):
    """Temporal CNN encoder, static metadata branch, and six-class head."""

    def __init__(self, *, telemetry_channels: int, metadata_dim: int, class_count: int = 6) -> None:
        super().__init__()
        if telemetry_channels <= 0 or metadata_dim <= 0 or class_count <= 1:
            raise ValueError("Model dimensions are invalid")
        self.telemetry_channels = telemetry_channels
        self.metadata_dim = metadata_dim
        self.class_count = class_count
        self.telemetry_encoder = nn.Sequential(
            nn.Conv1d(telemetry_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 12),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(36, 24),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(24, class_count),
        )

    def forward(self, telemetry: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if telemetry.ndim != 3 or telemetry.shape[2] != self.telemetry_channels:
            raise ValueError("telemetry must have shape [batch, time, channels]")
        if metadata.ndim != 2 or metadata.shape[1] != self.metadata_dim:
            raise ValueError("metadata must have shape [batch, metadata_dim]")
        if telemetry.shape[0] != metadata.shape[0]:
            raise ValueError("telemetry and metadata batch sizes must match")
        if not torch.isfinite(telemetry).all() or not torch.isfinite(metadata).all():
            raise ValueError("model inputs must be finite")
        encoded_telemetry = self.telemetry_encoder(telemetry.transpose(1, 2)).squeeze(-1)
        encoded_metadata = self.metadata_encoder(metadata)
        logits = self.classifier(torch.cat([encoded_telemetry, encoded_metadata], dim=1))
        if not torch.isfinite(logits).all():
            raise ValueError("model output must be finite")
        return logits
