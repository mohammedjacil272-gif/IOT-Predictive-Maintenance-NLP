"""Train the FD001 sensor-only LSTM baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_baseline import LSTMBaseline, count_parameters
from training.dataset import BaselineDataConfig, WindowDataset, load_train_val_data, prepare_baseline_data
from training.metrics import regression_metrics


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    input_size: int = 17
    sequence_length: int = 30
    hidden_size: int = 64
    num_layers: int = 2
    lstm_dropout: float = 0.2
    head_dropout: float = 0.2
    checkpoint_metric: str = "validation_rmse"
    checkpoint_mode: str = "min"
    data_dir: Path = Path("artifacts/data/fd001_lstm_baseline")
    model_dir: Path = Path("artifacts/models/fd001_lstm_baseline")
    metrics_dir: Path = Path("artifacts/metrics/fd001_lstm_baseline")
    embeddings_dir: Path = Path("artifacts/embeddings/fd001_lstm_baseline")

    def to_dict(self, device: str) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "early_stopping_patience": self.early_stopping_patience,
            "loss": "MSELoss",
            "optimizer": "Adam",
            "device": device,
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "lstm_dropout": self.lstm_dropout,
            "head_dropout": self.head_dropout,
            "checkpoint_metric": self.checkpoint_metric,
            "checkpoint_mode": self.checkpoint_mode,
            "official_test_evaluation": "not performed by training script",
            "data_dir": self.data_dir.as_posix(),
            "model_dir": self.model_dir.as_posix(),
            "metrics_dir": self.metrics_dir.as_posix(),
            "embeddings_dir": self.embeddings_dir.as_posix(),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            losses.append(float(loss.item()) * len(y_batch))
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    y_pred_np = np.concatenate(predictions)
    y_true_np = np.concatenate(targets)
    metrics = regression_metrics(y_true_np, y_pred_np)
    mean_loss = float(np.sum(losses) / len(y_true_np))
    return mean_loss, metrics, y_true_np, y_pred_np


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.MSELoss()
    losses: list[float] = []
    sample_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()) * len(y_batch))
        sample_count += len(y_batch)
    return float(np.sum(losses) / sample_count)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    parameter_count: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model_config,
            "training_config": training_config,
            "parameter_count": parameter_count,
        },
        path,
    )


def save_embeddings(
    model: LSTMBaseline,
    data_dir: Path,
    embeddings_dir: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    arrays = []
    metadata_frames = []
    for split_name, x_name, metadata_name in (
        ("train", "X_train.npy", "train_window_metadata.csv"),
        ("validation", "X_val.npy", "val_window_metadata.csv"),
    ):
        x_values = np.load(data_dir / x_name)
        metadata = pd.read_csv(data_dir / metadata_name)
        dataset = WindowDataset(x_values, np.zeros(len(x_values), dtype=np.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        split_embeddings: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for x_batch, _ in loader:
                embedding = model.extract_embedding(x_batch.to(device))
                split_embeddings.append(embedding.cpu().numpy())
        embedding_array = np.concatenate(split_embeddings).astype(np.float32)
        metadata = metadata.copy()
        offset = int(sum(len(array) for array in arrays))
        metadata["embedding_index"] = np.arange(offset, offset + len(metadata))
        metadata["embedding_source"] = split_name
        arrays.append(embedding_array)
        metadata_frames.append(metadata)
    all_embeddings = np.concatenate(arrays).astype(np.float32)
    all_metadata = pd.concat(metadata_frames, ignore_index=True)
    np.save(embeddings_dir / "sensor_embeddings.npy", all_embeddings)
    all_metadata.to_csv(embeddings_dir / "sensor_embedding_metadata.csv", index=False)


def run_training(config: TrainingConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    config.embeddings_dir.mkdir(parents=True, exist_ok=True)

    data_result = prepare_baseline_data(
        BaselineDataConfig(output_dir=config.data_dir, seed=config.seed)
    )
    X_train, y_train, X_val, y_val = load_train_val_data(config.data_dir)
    train_loader = DataLoader(
        WindowDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False)

    model = LSTMBaseline(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        lstm_dropout=config.lstm_dropout,
        head_dropout=config.head_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model_config = {
        "model": "LSTMBaseline",
        "input_size": config.input_size,
        "sequence_length": config.sequence_length,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "lstm_dropout": config.lstm_dropout,
        "head_dropout": config.head_dropout,
        "regression_head": ["Linear(64, 32)", "ReLU", "Dropout(0.2)", "Linear(32, 1)"],
        "output_activation": None,
        "embedding_dimension": config.hidden_size,
        "embedding_source": "final hidden state from last LSTM layer",
    }
    training_config = config.to_dict(str(device))
    parameter_count = count_parameters(model)
    model_config["parameter_count"] = parameter_count

    (config.model_dir / "model_config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    (config.model_dir / "training_config.json").write_text(json.dumps(training_config, indent=2, sort_keys=True), encoding="utf-8")
    (config.model_dir / "split_config.json").write_text(
        json.dumps(data_result["split_config"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_mae": val_metrics["mae"],
            "validation_rmse": val_metrics["rmse"],
            "validation_mse": val_metrics["mse"],
            "validation_r2": val_metrics["r2"],
        }
        history.append(row)
        improved = val_metrics["rmse"] < best_rmse
        if improved:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                config.model_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                model_config,
                training_config,
                parameter_count,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            config.model_dir / "last_model.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            model_config,
            training_config,
            parameter_count,
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f}"
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    duration_seconds = time.perf_counter() - start_time
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.metrics_dir / "training_history.csv", index=False)

    best_checkpoint = torch.load(config.model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    _, best_metrics, y_true, y_pred = evaluate_model(model, val_loader, device)
    validation_metrics = {
        **best_metrics,
        "best_epoch": int(best_epoch),
        "best_validation_rmse": float(best_rmse),
        "training_duration_seconds": float(duration_seconds),
        "epochs_completed": int(len(history_df)),
        "checkpoint_selection": "lowest validation RMSE",
        "official_test_evaluation": "not performed",
    }
    (config.metrics_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    predictions_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": y_pred - y_true})
    predictions_df.to_csv(config.metrics_dir / "validation_predictions.csv", index=False)
    save_embeddings(model, config.data_dir, config.embeddings_dir, device, config.batch_size)

    return {
        "data_result": data_result,
        "model_config": model_config,
        "training_config": training_config,
        "validation_metrics": validation_metrics,
        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
        },
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_rmse,
        "training_duration_seconds": duration_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FD001 LSTM baseline.")
    parser.add_argument("--data-dir", default="artifacts/data/fd001_lstm_baseline")
    parser.add_argument("--model-dir", default="artifacts/models/fd001_lstm_baseline")
    parser.add_argument("--metrics-dir", default="artifacts/metrics/fd001_lstm_baseline")
    parser.add_argument("--embeddings-dir", default="artifacts/embeddings/fd001_lstm_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        TrainingConfig(
            data_dir=Path(args.data_dir),
            model_dir=Path(args.model_dir),
            metrics_dir=Path(args.metrics_dir),
            embeddings_dir=Path(args.embeddings_dir),
        )
    )
    print("LSTM baseline training completed.")
    print(f"Shapes: {result['shapes']}")
    print(f"Train engines: {result['data_result']['split_config']['train_engine_count']}")
    print(f"Validation engines: {result['data_result']['split_config']['validation_engine_count']}")
    print(f"Model parameters: {result['parameter_count']}")
    print(f"Training duration seconds: {result['training_duration_seconds']:.2f}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best validation RMSE: {result['best_validation_rmse']:.6f}")
    print(f"Validation metrics: {result['validation_metrics']}")


if __name__ == "__main__":
    main()
