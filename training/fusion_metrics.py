"""Regression metrics for multimodal fusion evaluation."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and MSE for finite regression predictions."""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if true.shape != pred.shape:
        raise ValueError(
            f"Target and prediction shapes differ: {true.shape} vs {pred.shape}"
        )
    if true.size == 0:
        raise ValueError("Cannot evaluate empty arrays")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Targets and predictions must be finite")

    mse = float(np.mean((true - pred) ** 2))
    return {
        "MAE": float(np.mean(np.abs(true - pred))),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
    }
