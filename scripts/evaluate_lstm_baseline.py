"""Explicit official-test evaluation for the FD001 LSTM baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_baseline import LSTMBaseline
from training.dataset import WindowDataset
from training.metrics import regression_metrics


def evaluate_official_test(data_dir: Path, model_dir: Path, metrics_dir: Path, batch_size: int = 64) -> dict[str, float]:
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    metadata = pd.read_csv(data_dir / "test_window_metadata.csv")
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model = LSTMBaseline()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(WindowDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    predictions = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            predictions.append(y_pred.numpy())
            targets.append(y_batch.numpy())
    y_pred_np = np.concatenate(predictions)
    y_true_np = np.concatenate(targets)
    metrics = regression_metrics(y_true_np, y_pred_np)
    metrics.update(
        {
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "test_windows": int(len(X_test)),
            "official_test_labels_used": "final evaluation only",
        }
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output = metadata.copy()
    output["y_true"] = y_true_np
    output["y_pred"] = y_pred_np
    output["residual"] = y_pred_np - y_true_np
    output.to_csv(metrics_dir / "test_predictions.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FD001 LSTM baseline on official test set.")
    parser.add_argument("--data-dir", default="artifacts/data/fd001_lstm_baseline")
    parser.add_argument("--model-dir", default="artifacts/models/fd001_lstm_baseline")
    parser.add_argument("--metrics-dir", default="artifacts/metrics/fd001_lstm_baseline")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_official_test(
        Path(args.data_dir),
        Path(args.model_dir),
        Path(args.metrics_dir),
        batch_size=args.batch_size,
    )
    print("Official FD001 test evaluation completed.")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
