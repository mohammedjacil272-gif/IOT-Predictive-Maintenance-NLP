"""Multimodal sensor + maintenance-text fusion model."""

from __future__ import annotations

import torch
from torch import nn


class MultimodalFusionModel(nn.Module):
    """Predict RUL from 64-D sensor and 768-D text embeddings."""

    def __init__(
        self,
        sensor_dim: int = 64,
        text_dim: int = 768,
        hidden_dim: int = 128,
        head_dim: int = 64,
        dropout: float = 0.2,
        use_has_log: bool | None = None,
    ) -> None:
        super().__init__()

        if use_has_log is None:
            raise ValueError(
                "Select use_has_log explicitly; production policy is pending approval."
            )
        if min(sensor_dim, text_dim, hidden_dim, head_dim) <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.sensor_dim = sensor_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_has_log = use_has_log

        self.regression_head = nn.Sequential(
            nn.Linear(sensor_dim + text_dim + int(use_has_log), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        has_log: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sensor_embeddings.ndim != 2:
            raise ValueError("Sensor embeddings must have shape [batch, 64].")
        if text_embeddings.ndim != 2:
            raise ValueError("Text embeddings must have shape [batch, 768].")
        if sensor_embeddings.shape[1] != self.sensor_dim:
            raise ValueError(f"Expected {self.sensor_dim} sensor features.")
        if text_embeddings.shape[1] != self.text_dim:
            raise ValueError(f"Expected {self.text_dim} text features.")
        if sensor_embeddings.shape[0] != text_embeddings.shape[0]:
            raise ValueError("Sensor and text batch sizes must match.")
        if sensor_embeddings.device != text_embeddings.device:
            raise ValueError("Sensor and text tensors must use the same device.")
        if not torch.is_floating_point(sensor_embeddings):
            raise TypeError("Sensor embeddings must be floating-point tensors.")
        if not torch.is_floating_point(text_embeddings):
            raise TypeError("Text embeddings must be floating-point tensors.")
        if sensor_embeddings.dtype != text_embeddings.dtype:
            raise TypeError("Sensor and text tensors must use the same dtype.")
        if not torch.isfinite(sensor_embeddings).all():
            raise ValueError("Sensor embeddings contain NaN or Inf.")
        if not torch.isfinite(text_embeddings).all():
            raise ValueError("Text embeddings contain NaN or Inf.")

        parts = [sensor_embeddings, text_embeddings]
        if self.use_has_log:
            if has_log is None or has_log.shape != (sensor_embeddings.shape[0],):
                raise ValueError("has_log must have shape [batch] when enabled.")
            if has_log.device != sensor_embeddings.device:
                raise ValueError("has_log and embeddings must use the same device.")
            parts.append(has_log.to(dtype=sensor_embeddings.dtype).unsqueeze(1))
        fused = torch.cat(parts, dim=1)
        output = self.regression_head(fused).squeeze(-1)
        if not torch.isfinite(output).all().item():
            raise ValueError("Fusion model output contains NaN or Inf.")
        return output
