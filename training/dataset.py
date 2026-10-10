"""Data preparation and PyTorch datasets for the FD001 LSTM baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from preprocessing.cmapss_fd001 import (
    CANONICAL_COLUMNS,
    EXPECTED_ENGINES,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    SELECTED_FEATURES,
)


WINDOW_SIZE = 30
STRIDE = 1
SEED = 42


@dataclass(frozen=True)
class BaselineDataConfig:
    train_path: Path = Path("Data/train_FD001.txt")
    test_path: Path = Path("Data/test_FD001.txt")
    rul_path: Path = Path("Data/RUL_FD001.txt")
    output_dir: Path = Path("artifacts/data/fd001_lstm_baseline")
    seed: int = SEED
    window_size: int = WINDOW_SIZE
    stride: int = STRIDE
    train_engine_count: int = 80
    val_engine_count: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_path": self.train_path.as_posix(),
            "test_path": self.test_path.as_posix(),
            "rul_path": self.rul_path.as_posix(),
            "output_dir": self.output_dir.as_posix(),
            "seed": self.seed,
            "window_size": self.window_size,
            "stride": self.stride,
            "train_engine_count": self.train_engine_count,
            "val_engine_count": self.val_engine_count,
            "selected_features": SELECTED_FEATURES,
            "scaler": "StandardScaler",
            "scaler_fit_scope": "raw rows from the 80 training engines only",
            "official_test_usage": "prepared for explicit final evaluation only; not used in training or model selection",
        }


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def read_raw_fd001(config: BaselineDataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (config.train_path, config.test_path, config.rul_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required FD001 file: {path}")
    train_df = pd.read_csv(config.train_path, sep=r"\s+", header=None, names=CANONICAL_COLUMNS, engine="python")
    test_df = pd.read_csv(config.test_path, sep=r"\s+", header=None, names=CANONICAL_COLUMNS, engine="python")
    rul_df = pd.read_csv(config.rul_path, sep=r"\s+", header=None, names=["final_rul"], engine="python")
    return train_df, test_df, rul_df


def validate_raw_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> None:
    if train_df.shape != (EXPECTED_TRAIN_ROWS, len(CANONICAL_COLUMNS)):
        raise ValueError(f"Unexpected train shape: {train_df.shape}")
    if test_df.shape != (EXPECTED_TEST_ROWS, len(CANONICAL_COLUMNS)):
        raise ValueError(f"Unexpected test shape: {test_df.shape}")
    if rul_df.shape != (EXPECTED_ENGINES, 1):
        raise ValueError(f"Unexpected RUL shape: {rul_df.shape}")
    for name, df in (("train", train_df), ("test", test_df)):
        if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} contains missing, NaN, or Inf values")
        unit_ids = sorted(df["unit_id"].astype(int).unique().tolist())
        if unit_ids != list(range(1, EXPECTED_ENGINES + 1)):
            raise ValueError(f"{name} engine IDs are not 1..100")
        if df.duplicated(["unit_id", "cycle"]).any():
            raise ValueError(f"{name} has duplicate unit/cycle rows")
        for unit_id, group in df.groupby("unit_id", sort=True):
            cycles = group["cycle"].astype(int).tolist()
            if cycles != list(range(1, len(cycles) + 1)):
                raise ValueError(f"{name} cycles are not sequential for unit {unit_id}")


def split_engine_ids(seed: int) -> tuple[list[int], list[int]]:
    engine_ids = np.arange(1, EXPECTED_ENGINES + 1)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engine_ids)
    train_ids = sorted(int(engine_id) for engine_id in shuffled[:80])
    val_ids = sorted(int(engine_id) for engine_id in shuffled[80:])
    return train_ids, val_ids


def add_rul_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    train["max_cycle"] = train.groupby("unit_id")["cycle"].transform("max")
    train["RUL"] = train["max_cycle"] - train["cycle"]

    rul = rul_df.copy()
    rul["unit_id"] = np.arange(1, len(rul) + 1)
    test = test.merge(rul[["unit_id", "final_rul"]], on="unit_id", how="left", validate="many_to_one")
    test["last_observed_cycle"] = test.groupby("unit_id")["cycle"].transform("max")
    test["row_RUL"] = test["final_rul"] + (test["last_observed_cycle"] - test["cycle"])
    test["RUL"] = test["row_RUL"]
    return train, test


def scale_by_train_engines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_engine_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, dict[str, Any]]:
    scaler_fit_df = train_df[train_df["unit_id"].isin(train_engine_ids)]
    scaler = StandardScaler()
    scaler.fit(scaler_fit_df[SELECTED_FEATURES].to_numpy(dtype=np.float64))

    scaled_train = train_df.copy()
    scaled_test = test_df.copy()
    train_scaled_values = scaler.transform(train_df[SELECTED_FEATURES].to_numpy(dtype=np.float64))
    test_scaled_values = scaler.transform(test_df[SELECTED_FEATURES].to_numpy(dtype=np.float64))
    for feature_index, feature in enumerate(SELECTED_FEATURES):
        scaled_train[feature] = train_scaled_values[:, feature_index]
        scaled_test[feature] = test_scaled_values[:, feature_index]
    report = {
        "scaler": "StandardScaler",
        "fit_scope": "training_engine_rows_only",
        "fit_engine_ids": train_engine_ids,
        "fit_rows": int(len(scaler_fit_df)),
        "fit_feature_count": len(SELECTED_FEATURES),
        "feature_order": SELECTED_FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    return scaled_train, scaled_test, scaler, report


def _windows_for_engines(
    df: pd.DataFrame,
    engine_ids: list[int],
    source_split: str,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []
    for unit_id in engine_ids:
        group = df[df["unit_id"] == unit_id].sort_values("cycle").reset_index(drop=True)
        features = group[SELECTED_FEATURES].to_numpy(dtype=np.float32)
        cycles = group["cycle"].astype(int).to_numpy()
        ruls = group["RUL"].astype(float).to_numpy()
        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            window_cycles = cycles[start:end]
            if not np.array_equal(window_cycles, np.arange(window_cycles[0], window_cycles[0] + window_size)):
                raise ValueError(f"Window cycles are not consecutive for {source_split} unit {unit_id}")
            label_rul = float(ruls[end - 1])
            windows.append(features[start:end])
            labels.append(label_rul)
            metadata.append(
                {
                    "unit_id": int(unit_id),
                    "window_start_cycle": int(window_cycles[0]),
                    "window_end_cycle": int(window_cycles[-1]),
                    "label_cycle": int(window_cycles[-1]),
                    "label_rul": label_rul,
                    "source_split": source_split,
                }
            )
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.float32), pd.DataFrame(metadata)


def _terminal_test_windows(
    test_df: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []
    for unit_id, group in test_df.groupby("unit_id", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        terminal = group.iloc[-window_size:]
        cycles = terminal["cycle"].astype(int).to_numpy()
        if len(terminal) != window_size:
            raise ValueError(f"Test unit {unit_id} has fewer than {window_size} cycles")
        if not np.array_equal(cycles, np.arange(cycles[0], cycles[0] + window_size)):
            raise ValueError(f"Terminal test cycles are not consecutive for unit {unit_id}")
        final_rul = float(group["final_rul"].iloc[-1])
        windows.append(terminal[SELECTED_FEATURES].to_numpy(dtype=np.float32))
        labels.append(final_rul)
        metadata.append(
            {
                "unit_id": int(unit_id),
                "window_start_cycle": int(cycles[0]),
                "window_end_cycle": int(cycles[-1]),
                "label_cycle": int(cycles[-1]),
                "label_rul": final_rul,
                "source_split": "official_test",
                "final_rul_from_RUL_file": final_rul,
            }
        )
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.float32), pd.DataFrame(metadata)


def prepare_baseline_data(config: BaselineDataConfig) -> dict[str, Any]:
    train_raw, test_raw, rul_df = read_raw_fd001(config)
    validate_raw_frames(train_raw, test_raw, rul_df)
    train_engine_ids, val_engine_ids = split_engine_ids(config.seed)
    if set(train_engine_ids) & set(val_engine_ids):
        raise ValueError("Train/validation engine split overlaps")

    train_with_rul, test_with_rul = add_rul_columns(train_raw, test_raw, rul_df)
    scaled_train, scaled_test, scaler, scaler_report = scale_by_train_engines(
        train_with_rul,
        test_with_rul,
        train_engine_ids,
    )
    X_train, y_train, train_metadata = _windows_for_engines(
        scaled_train,
        train_engine_ids,
        "train",
        config.window_size,
        config.stride,
    )
    X_val, y_val, val_metadata = _windows_for_engines(
        scaled_train,
        val_engine_ids,
        "validation",
        config.window_size,
        config.stride,
    )
    X_test, y_test, test_metadata = _terminal_test_windows(scaled_test, config.window_size)

    if X_train.shape[1:] != (config.window_size, len(SELECTED_FEATURES)):
        raise ValueError(f"Unexpected train window shape: {X_train.shape}")
    if X_val.shape[1:] != (config.window_size, len(SELECTED_FEATURES)):
        raise ValueError(f"Unexpected validation window shape: {X_val.shape}")
    if X_test.shape != (EXPECTED_ENGINES, config.window_size, len(SELECTED_FEATURES)):
        raise ValueError(f"Unexpected official test shape: {X_test.shape}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.output_dir / "scaler.joblib")
    np.save(config.output_dir / "X_train.npy", X_train)
    np.save(config.output_dir / "y_train.npy", y_train)
    np.save(config.output_dir / "X_val.npy", X_val)
    np.save(config.output_dir / "y_val.npy", y_val)
    np.save(config.output_dir / "X_test.npy", X_test)
    np.save(config.output_dir / "y_test.npy", y_test)
    train_metadata.to_csv(config.output_dir / "train_window_metadata.csv", index=False)
    val_metadata.to_csv(config.output_dir / "val_window_metadata.csv", index=False)
    test_metadata.to_csv(config.output_dir / "test_window_metadata.csv", index=False)

    split_config = {
        "seed": config.seed,
        "split_unit": "engine_id from raw FD001 training file",
        "train_engine_ids": train_engine_ids,
        "validation_engine_ids": val_engine_ids,
        "train_engine_count": len(train_engine_ids),
        "validation_engine_count": len(val_engine_ids),
        "overlap": sorted(set(train_engine_ids) & set(val_engine_ids)),
        "official_test_set": "Data/test_FD001.txt isolated from training and model selection",
    }
    baseline_config = config.to_dict()
    data_report = {
        "X_train_shape": list(X_train.shape),
        "y_train_shape": list(y_train.shape),
        "X_val_shape": list(X_val.shape),
        "y_val_shape": list(y_val.shape),
        "X_test_shape": list(X_test.shape),
        "y_test_shape": list(y_test.shape),
        "finite_values": bool(
            np.isfinite(X_train).all()
            and np.isfinite(y_train).all()
            and np.isfinite(X_val).all()
            and np.isfinite(y_val).all()
            and np.isfinite(X_test).all()
            and np.isfinite(y_test).all()
        ),
        "scaler_report": scaler_report,
        "test_labels_usage": "saved for explicit final evaluation only; not used during training",
    }
    (config.output_dir / "baseline_preprocessing_config.json").write_text(
        json.dumps(baseline_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (config.output_dir / "split_config.json").write_text(
        json.dumps(split_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (config.output_dir / "data_report.json").write_text(
        json.dumps(data_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "config": baseline_config,
        "split_config": split_config,
        "data_report": data_report,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_metadata": train_metadata,
        "val_metadata": val_metadata,
        "test_metadata": test_metadata,
    }


def load_train_val_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(data_dir / "X_train.npy"),
        np.load(data_dir / "y_train.npy"),
        np.load(data_dir / "X_val.npy"),
        np.load(data_dir / "y_val.npy"),
    )
