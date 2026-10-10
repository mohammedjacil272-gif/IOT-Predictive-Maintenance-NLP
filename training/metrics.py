"""Regression metrics for RUL prediction."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual = float(np.sum(errors**2))
    r2 = float(1.0 - residual / total) if total > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}
