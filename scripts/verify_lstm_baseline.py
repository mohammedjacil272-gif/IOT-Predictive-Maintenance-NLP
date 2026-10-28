"""Verify FD001 LSTM baseline artifacts and leakage controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_baseline import LSTMBaseline, count_parameters
from preprocessing.cmapss_fd001 import CANONICAL_COLUMNS


REQUIRED_DATA_FILES = [
    "baseline_preprocessing_config.json",
    "split_config.json",
    "data_report.json",
    "scaler.joblib",
    "X_train.npy",
    "y_train.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test.npy",
    "y_test.npy",
    "train_window_metadata.csv",
    "val_window_metadata.csv",
    "test_window_metadata.csv",
]

REQUIRED_TRAINED_FILES = [
    "best_model.pt",
    "last_model.pt",
    "model_config.json",
    "split_config.json",
    "training_config.json",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_data_artifacts(data_dir: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing baseline data artifacts: {missing}")

    split_config = load_json(data_dir / "split_config.json")
    data_report = load_json(data_dir / "data_report.json")
    scaler = joblib.load(data_dir / "scaler.joblib")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    train_metadata = pd.read_csv(data_dir / "train_window_metadata.csv")
    val_metadata = pd.read_csv(data_dir / "val_window_metadata.csv")
    test_metadata = pd.read_csv(data_dir / "test_window_metadata.csv")
    raw_train = pd.read_csv("Data/train_FD001.txt", sep=r"\s+", header=None, names=CANONICAL_COLUMNS, engine="python")
    expected_scaler_fit_rows = int(raw_train[raw_train["unit_id"].isin(split_config["train_engine_ids"])].shape[0])

    train_ids = set(split_config["train_engine_ids"])
    val_ids = set(split_config["validation_engine_ids"])
    checks: dict[str, object] = {
        "data_artifacts_exist": True,
        "train_engine_count_is_80": len(train_ids) == 80,
        "validation_engine_count_is_20": len(val_ids) == 20,
        "train_validation_engine_overlap_zero": len(train_ids & val_ids) == 0,
        "official_test_source_is_isolated": set(test_metadata["source_split"].unique()) == {"official_test"},
        "train_metadata_source": set(train_metadata["source_split"].unique()) == {"train"},
        "validation_metadata_source": set(val_metadata["source_split"].unique()) == {"validation"},
        "scaler_is_standard_scaler": scaler.__class__.__name__ == "StandardScaler",
        "scaler_fit_rows_match_training_engines": data_report["scaler_report"]["fit_rows"]
        == expected_scaler_fit_rows,
        "scaler_fit_feature_count_is_17": int(scaler.n_features_in_) == 17,
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
        "input_shape_train_is_30_by_17": X_train.shape[1:] == (30, 17),
        "input_shape_validation_is_30_by_17": X_val.shape[1:] == (30, 17),
        "input_shape_test_is_100_30_17": X_test.shape == (100, 30, 17),
        "y_test_shape_is_100": y_test.shape == (100,),
        "train_lengths_match": len(X_train) == len(y_train) == len(train_metadata),
        "validation_lengths_match": len(X_val) == len(y_val) == len(val_metadata),
        "test_lengths_match": len(X_test) == len(y_test) == len(test_metadata),
        "finite_values": bool(
            np.isfinite(X_train).all()
            and np.isfinite(y_train).all()
            and np.isfinite(X_val).all()
            and np.isfinite(y_val).all()
            and np.isfinite(X_test).all()
            and np.isfinite(y_test).all()
        ),
        "test_labels_not_marked_for_training": data_report["test_labels_usage"].startswith("saved for explicit final evaluation only"),
        "deterministic_seed_recorded": split_config["seed"] == 42,
        "window_lengths_are_30": bool(
            ((train_metadata["window_end_cycle"] - train_metadata["window_start_cycle"] + 1) == 30).all()
            and ((val_metadata["window_end_cycle"] - val_metadata["window_start_cycle"] + 1) == 30).all()
            and ((test_metadata["window_end_cycle"] - test_metadata["window_start_cycle"] + 1) == 30).all()
        ),
    }
    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    if failed:
        raise ValueError(f"Data verification failed: {failed}")
    return checks


def verify_trained_artifacts(
    data_dir: Path,
    model_dir: Path,
    metrics_dir: Path,
    embeddings_dir: Path,
) -> dict[str, object]:
    missing = [name for name in REQUIRED_TRAINED_FILES if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained baseline artifacts; verifier is read-only and will "
            f"not regenerate protected files: {missing}"
        )
    checks = verify_data_artifacts(data_dir)

    history = pd.read_csv(metrics_dir / "training_history.csv")
    validation_metrics = load_json(metrics_dir / "validation_metrics.json")
    model_config = load_json(model_dir / "model_config.json")
    training_config = load_json(model_dir / "training_config.json")
    best_checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(model_dir / "last_model.pt", map_location="cpu", weights_only=False)
    model = LSTMBaseline()
    expected_parameter_count = count_parameters(model)
    lowest_rmse_row = history.loc[history["validation_rmse"].idxmin()]

    embeddings = np.load(embeddings_dir / "sensor_embeddings.npy")
    embedding_metadata = pd.read_csv(embeddings_dir / "sensor_embedding_metadata.csv")
    train_metadata = pd.read_csv(data_dir / "train_window_metadata.csv")
    val_metadata = pd.read_csv(data_dir / "val_window_metadata.csv")

    trained_checks: dict[str, object] = {
        "trained_artifacts_exist": True,
        "model_parameter_count_matches": model_config["parameter_count"] == expected_parameter_count,
        "checkpoint_parameter_count_matches": best_checkpoint["parameter_count"] == expected_parameter_count,
        "training_seed_is_42": training_config["seed"] == 42,
        "training_config_official_test_not_evaluated": training_config["official_test_evaluation"] == "not performed by training script",
        "best_checkpoint_lowest_validation_rmse": int(best_checkpoint["epoch"]) == int(lowest_rmse_row["epoch"]),
        "best_metric_matches_history": abs(float(best_checkpoint["metrics"]["rmse"]) - float(lowest_rmse_row["validation_rmse"])) < 1e-6,
        "validation_metrics_match_best_epoch": int(validation_metrics["best_epoch"]) == int(lowest_rmse_row["epoch"]),
        "last_checkpoint_exists": (model_dir / "last_model.pt").exists() and int(last_checkpoint["epoch"]) == int(history["epoch"].max()),
        "test_metrics_not_created_by_training": not (metrics_dir / "test_metrics.json").exists(),
        "embeddings_shape": list(embeddings.shape),
        "embedding_count_matches_train_plus_validation": len(embeddings)
        == len(train_metadata) + len(val_metadata)
        == len(embedding_metadata),
        "embedding_dimension_is_64": embeddings.shape[1] == 64,
        "embeddings_finite": bool(np.isfinite(embeddings).all()),
    }
    checks.update(trained_checks)
    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    if failed:
        raise ValueError(f"LSTM baseline verification failed: {failed}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FD001 LSTM baseline artifacts.")
    parser.add_argument("--data-dir", default="artifacts/data/fd001_lstm_baseline")
    parser.add_argument("--model-dir", default="artifacts/models/fd001_lstm_baseline")
    parser.add_argument("--metrics-dir", default="artifacts/metrics/fd001_lstm_baseline")
    parser.add_argument("--embeddings-dir", default="artifacts/embeddings/fd001_lstm_baseline")
    parser.add_argument("--pretrain", action="store_true", help="Verify existing data artifacts only; never create them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.pretrain:
        checks = verify_data_artifacts(data_dir)
        print("Pre-training LSTM baseline data verification passed.")
    else:
        checks = verify_trained_artifacts(
            data_dir,
            Path(args.model_dir),
            Path(args.metrics_dir),
            Path(args.embeddings_dir),
        )
        print("LSTM baseline verification passed.")
    for name, value in checks.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
