"""Leakage-safe Azure PdM telemetry aggregation features."""

from __future__ import annotations

import numpy as np

from .train import TELEMETRY_FIELDS


def feature_names(*, engineered: bool, include_metadata: bool) -> list[str]:
    names: list[str] = []
    if engineered:
        statistics = ("mean", "std", "min", "max", "range", "first_last_delta", "slope", "mean_abs_diff")
    else:
        statistics = ("mean", "std")
    for field in TELEMETRY_FIELDS:
        names.extend(f"{field}_{stat}" for stat in statistics)
    if include_metadata:
        names.append("machine_age_standardized")
        names.append("machine_model_one_hot")
    return names


def build_feature_matrix(telemetry: np.ndarray, metadata: np.ndarray, *, engineered: bool, include_metadata: bool) -> np.ndarray:
    if telemetry.ndim != 3 or telemetry.shape[2] != len(TELEMETRY_FIELDS):
        raise ValueError("telemetry must have shape [samples, time, 4]")
    if metadata.ndim != 2 or metadata.shape[0] != telemetry.shape[0]:
        raise ValueError("metadata must align with telemetry rows")
    mean = telemetry.mean(axis=1)
    std = telemetry.std(axis=1)
    blocks = [mean, std]
    if engineered:
        minimum = telemetry.min(axis=1)
        maximum = telemetry.max(axis=1)
        delta = telemetry[:, -1, :] - telemetry[:, 0, :]
        denominator = max(telemetry.shape[1] - 1, 1)
        slope = delta / denominator
        mean_abs_diff = np.abs(np.diff(telemetry, axis=1)).mean(axis=1) if telemetry.shape[1] > 1 else np.zeros_like(mean)
        blocks.extend([minimum, maximum, maximum - minimum, delta, slope, mean_abs_diff])
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    if include_metadata:
        output = np.concatenate([output, metadata], axis=1)
    if not np.isfinite(output).all():
        raise ValueError("engineered features must be finite")
    return output
