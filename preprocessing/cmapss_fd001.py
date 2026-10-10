"""C-MAPSS FD001 preprocessing pipeline.

This module builds model-ready arrays for the future sensor-only LSTM baseline.
It does not train models and does not modify the source dataset files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CANONICAL_COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

RAW_AVAILABLE_FEATURES = [
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

SELECTED_FEATURES = [
    "setting_1",
    "setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

EXCLUDED_CONSTANT_FEATURES = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

EXPECTED_TRAIN_ROWS = 20631
EXPECTED_TEST_ROWS = 13096
EXPECTED_ENGINES = 100
EXPECTED_COLUMNS = 26
WINDOW_SIZE = 30
STRIDE = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    train_path: Path = Path("Data/train_FD001.txt")
    test_path: Path = Path("Data/test_FD001.txt")
    rul_path: Path = Path("Data/RUL_FD001.txt")
    output_dir: Path = Path("artifacts/preprocessing/fd001")
    window_size: int = WINDOW_SIZE
    stride: int = STRIDE
    scaler: str = "StandardScaler"

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_path": self.train_path.as_posix(),
            "test_path": self.test_path.as_posix(),
            "rul_path": self.rul_path.as_posix(),
            "output_dir": self.output_dir.as_posix(),
            "window_size": self.window_size,
            "stride": self.stride,
            "scaler": self.scaler,
            "rul_target": "uncapped",
            "test_window_policy": "one_terminal_window_per_engine",
            "unit_id_and_cycle": "metadata_only",
            "canonical_columns": CANONICAL_COLUMNS,
            "raw_available_features": RAW_AVAILABLE_FEATURES,
            "selected_features": SELECTED_FEATURES,
            "excluded_constant_features": EXCLUDED_CONSTANT_FEATURES,
        }


def load_fd001_files(config: PreprocessingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (config.train_path, config.test_path, config.rul_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    train_df = pd.read_csv(
        config.train_path,
        sep=r"\s+",
        header=None,
        names=CANONICAL_COLUMNS,
        engine="python",
    )
    test_df = pd.read_csv(
        config.test_path,
        sep=r"\s+",
        header=None,
        names=CANONICAL_COLUMNS,
        engine="python",
    )
    rul_df = pd.read_csv(
        config.rul_path,
        sep=r"\s+",
        header=None,
        names=["final_rul"],
        engine="python",
    )
    return train_df, test_df, rul_df


def _ensure_finite(df: pd.DataFrame, name: str) -> None:
    values = df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _validate_engine_cycles(df: pd.DataFrame, name: str) -> None:
    unit_ids = sorted(df["unit_id"].astype(int).unique().tolist())
    expected_ids = list(range(1, EXPECTED_ENGINES + 1))
    if unit_ids != expected_ids:
        raise ValueError(f"{name} engine IDs are not sequential 1..{EXPECTED_ENGINES}")

    duplicate_cycles = df.duplicated(subset=["unit_id", "cycle"]).sum()
    if duplicate_cycles:
        raise ValueError(f"{name} has {duplicate_cycles} duplicate unit/cycle rows")

    for unit_id, group in df.groupby("unit_id", sort=True):
        cycles = group["cycle"].astype(int).tolist()
        expected_cycles = list(range(1, len(cycles) + 1))
        if cycles != expected_cycles:
            raise ValueError(f"{name} cycles are not sequential for unit_id={unit_id}")


def validate_raw_data(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    if train_df.shape != (EXPECTED_TRAIN_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"Unexpected train shape: {train_df.shape}")
    if test_df.shape != (EXPECTED_TEST_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"Unexpected test shape: {test_df.shape}")
    if rul_df.shape != (EXPECTED_ENGINES, 1):
        raise ValueError(f"Unexpected RUL shape: {rul_df.shape}")

    if list(train_df.columns) != CANONICAL_COLUMNS or list(test_df.columns) != CANONICAL_COLUMNS:
        raise ValueError("Canonical column assignment failed")
    if train_df.isna().any().any() or test_df.isna().any().any() or rul_df.isna().any().any():
        raise ValueError("Missing values found in raw FD001 files")

    _ensure_finite(train_df, "train_FD001")
    _ensure_finite(test_df, "test_FD001")
    _ensure_finite(rul_df, "RUL_FD001")
    _validate_engine_cycles(train_df, "train_FD001")
    _validate_engine_cycles(test_df, "test_FD001")

    checks["input_files_exist"] = True
    checks["train_shape"] = list(train_df.shape)
    checks["test_shape"] = list(test_df.shape)
    checks["rul_shape"] = list(rul_df.shape)
    checks["columns"] = CANONICAL_COLUMNS
    checks["train_engines"] = int(train_df["unit_id"].nunique())
    checks["test_engines"] = int(test_df["unit_id"].nunique())
    checks["rul_values"] = int(len(rul_df))
    checks["engine_ids_1_to_100"] = True
    checks["sequential_cycles"] = True
    checks["duplicate_cycles"] = 0
    checks["missing_values"] = 0
    checks["raw_values_finite"] = True
    return checks


def add_rul_targets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    train["max_cycle"] = train.groupby("unit_id")["cycle"].transform("max")
    train["RUL"] = train["max_cycle"] - train["cycle"]

    rul = rul_df.copy()
    rul["unit_id"] = np.arange(1, len(rul) + 1)
    rul = rul[["unit_id", "final_rul"]]

    test = test.merge(rul, on="unit_id", how="left", validate="many_to_one")
    if test["final_rul"].isna().any():
        raise ValueError("Failed to align RUL values to all test engines")

    test["last_observed_cycle"] = test.groupby("unit_id")["cycle"].transform("max")
    test["row_RUL"] = test["final_rul"] + (test["last_observed_cycle"] - test["cycle"])
    test["RUL"] = test["row_RUL"]

    if int(test.groupby("unit_id")["final_rul"].first().shape[0]) != EXPECTED_ENGINES:
        raise ValueError("RUL-engine alignment did not produce 100 test engines")

    return train, test


def build_feature_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    train_constant = [
        feature for feature in RAW_AVAILABLE_FEATURES if train_df[feature].nunique(dropna=False) == 1
    ]
    test_constant = [
        feature for feature in RAW_AVAILABLE_FEATURES if test_df[feature].nunique(dropna=False) == 1
    ]
    candidate_informative = [
        feature for feature in RAW_AVAILABLE_FEATURES if feature not in EXCLUDED_CONSTANT_FEATURES
    ]
    return {
        "raw_available_features": RAW_AVAILABLE_FEATURES,
        "candidate_informative_features": candidate_informative,
        "final_selected_features": SELECTED_FEATURES,
        "excluded_constant_features": EXCLUDED_CONSTANT_FEATURES,
        "train_constant_features_detected": train_constant,
        "test_constant_features_detected": test_constant,
        "unit_id_and_cycle": "metadata_only",
        "feature_count": len(SELECTED_FEATURES),
        "selection_rule": "approved_17_documented_non_constant_features",
    }


def fit_and_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, dict[str, Any]]:
    if len(SELECTED_FEATURES) != 17:
        raise ValueError("Expected exactly 17 selected features")
    missing_features = sorted(set(SELECTED_FEATURES) - set(train_df.columns))
    if missing_features:
        raise ValueError(f"Missing selected features: {missing_features}")
    if "unit_id" in SELECTED_FEATURES or "cycle" in SELECTED_FEATURES:
        raise ValueError("unit_id and cycle must not be model features")

    scaler = StandardScaler()
    train_values = train_df[SELECTED_FEATURES].to_numpy(dtype=np.float64)
    test_values = test_df[SELECTED_FEATURES].to_numpy(dtype=np.float64)
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    train_scaled_df = train_df[["unit_id", "cycle", "RUL"]].copy()
    test_scaled_df = test_df[
        ["unit_id", "cycle", "RUL", "row_RUL", "final_rul", "last_observed_cycle"]
    ].copy()

    train_scaled_df.loc[:, SELECTED_FEATURES] = train_scaled
    test_scaled_df.loc[:, SELECTED_FEATURES] = test_scaled

    scaler_report = {
        "scaler": "StandardScaler",
        "fitted_on": "training_rows_only",
        "fit_rows": int(train_values.shape[0]),
        "fit_feature_count": int(train_values.shape[1]),
        "feature_order": SELECTED_FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    return train_scaled_df, test_scaled_df, scaler, scaler_report


def generate_train_windows(
    train_df: pd.DataFrame,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []

    for unit_id, group in train_df.groupby("unit_id", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        feature_values = group[SELECTED_FEATURES].to_numpy(dtype=np.float32)
        cycles = group["cycle"].astype(int).to_numpy()
        ruls = group["RUL"].astype(float).to_numpy()
        for start in range(0, len(group) - window_size + 1, stride):
            end = start + window_size
            window_cycles = cycles[start:end]
            if not np.array_equal(window_cycles, np.arange(window_cycles[0], window_cycles[0] + window_size)):
                raise ValueError(f"Non-consecutive train window for unit_id={unit_id}")
            label_rul = ruls[end - 1]
            windows.append(feature_values[start:end])
            labels.append(label_rul)
            metadata.append(
                {
                    "unit_id": int(unit_id),
                    "window_start_cycle": int(window_cycles[0]),
                    "window_end_cycle": int(window_cycles[-1]),
                    "label_cycle": int(window_cycles[-1]),
                    "label_rul": float(label_rul),
                    "source_split": "train",
                }
            )

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def generate_test_windows(
    test_df: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []

    for unit_id, group in test_df.groupby("unit_id", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        if len(group) < window_size:
            raise ValueError(f"Test unit_id={unit_id} has fewer than {window_size} cycles")
        terminal = group.iloc[-window_size:]
        cycles = terminal["cycle"].astype(int).to_numpy()
        if not np.array_equal(cycles, np.arange(cycles[0], cycles[0] + window_size)):
            raise ValueError(f"Non-consecutive terminal test window for unit_id={unit_id}")
        final_rul = float(group["final_rul"].iloc[-1])
        row_rul_at_terminal = float(group["row_RUL"].iloc[-1])
        if final_rul != row_rul_at_terminal:
            raise ValueError(f"Terminal row RUL does not match final RUL for unit_id={unit_id}")

        windows.append(terminal[SELECTED_FEATURES].to_numpy(dtype=np.float32))
        labels.append(final_rul)
        metadata.append(
            {
                "unit_id": int(unit_id),
                "window_start_cycle": int(cycles[0]),
                "window_end_cycle": int(cycles[-1]),
                "label_cycle": int(cycles[-1]),
                "label_rul": final_rul,
                "source_split": "test",
                "final_rul_from_RUL_file": final_rul,
            }
        )

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def validate_outputs(
    config: PreprocessingConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    scaler_report: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["selected_feature_count"] = len(SELECTED_FEATURES)
    checks["selected_feature_count_is_17"] = len(SELECTED_FEATURES) == 17
    checks["scaler_fitted_only_on_training_rows"] = scaler_report["fit_rows"] == len(train_df)
    checks["X_train_shape"] = list(X_train.shape)
    checks["y_train_shape"] = list(y_train.shape)
    checks["X_test_shape"] = list(X_test.shape)
    checks["y_test_shape"] = list(y_test.shape)
    checks["X_train_finite"] = bool(np.isfinite(X_train).all())
    checks["y_train_finite"] = bool(np.isfinite(y_train).all())
    checks["X_test_finite"] = bool(np.isfinite(X_test).all())
    checks["y_test_finite"] = bool(np.isfinite(y_test).all())
    checks["train_lengths_match"] = len(X_train) == len(y_train) == len(train_metadata)
    checks["test_lengths_match"] = len(X_test) == len(y_test) == len(test_metadata)
    checks["test_window_count_is_100"] = len(X_test) == EXPECTED_ENGINES
    checks["expected_test_shape"] = list(X_test.shape) == [100, config.window_size, len(SELECTED_FEATURES)]
    checks["expected_y_test_shape"] = list(y_test.shape) == [100]
    checks["train_window_width"] = X_train.shape[1:] == (config.window_size, len(SELECTED_FEATURES))
    checks["test_window_width"] = X_test.shape[1:] == (config.window_size, len(SELECTED_FEATURES))

    train_labels_ok = True
    for row in train_metadata.itertuples(index=False):
        expected_rul = train_df.loc[
            (train_df["unit_id"] == row.unit_id) & (train_df["cycle"] == row.label_cycle),
            "RUL",
        ].iloc[0]
        if float(expected_rul) != float(row.label_rul):
            train_labels_ok = False
            break
    checks["training_labels_equal_final_cycle_rul"] = train_labels_ok

    test_labels_ok = np.array_equal(
        y_test.astype(np.float32),
        rul_df["final_rul"].to_numpy(dtype=np.float32),
    )
    metadata_test_labels_ok = np.array_equal(
        test_metadata["final_rul_from_RUL_file"].to_numpy(dtype=np.float32),
        rul_df["final_rul"].to_numpy(dtype=np.float32),
    )
    checks["test_labels_equal_rul_file"] = bool(test_labels_ok and metadata_test_labels_ok)

    train_cycles_ok = ((train_metadata["window_end_cycle"] - train_metadata["window_start_cycle"] + 1) == config.window_size).all()
    test_cycles_ok = ((test_metadata["window_end_cycle"] - test_metadata["window_start_cycle"] + 1) == config.window_size).all()
    checks["every_window_has_30_consecutive_cycles"] = bool(train_cycles_ok and test_cycles_ok)
    checks["no_windows_cross_engine_boundaries"] = True
    checks["rul_engine_alignment"] = len(rul_df) == test_df["unit_id"].nunique() == EXPECTED_ENGINES
    checks["uncapped_rul"] = int(train_df["RUL"].max()) > 125

    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    if failed:
        raise ValueError(f"Validation failed: {failed}")
    return checks


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifacts(
    config: PreprocessingConfig,
    scaler: StandardScaler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    feature_report: dict[str, Any],
    validation_report: dict[str, Any],
) -> dict[str, str]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "preprocessing_config": config.output_dir / "preprocessing_config.json",
        "feature_report": config.output_dir / "feature_report.json",
        "scaler": config.output_dir / "scaler.joblib",
        "X_train": config.output_dir / "X_train.npy",
        "y_train": config.output_dir / "y_train.npy",
        "X_test": config.output_dir / "X_test.npy",
        "y_test": config.output_dir / "y_test.npy",
        "train_window_metadata": config.output_dir / "train_window_metadata.csv",
        "test_window_metadata": config.output_dir / "test_window_metadata.csv",
        "validation_report": config.output_dir / "validation_report.json",
    }

    paths["preprocessing_config"].write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["feature_report"].write_text(
        json.dumps(feature_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    joblib.dump(scaler, paths["scaler"])
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_test"], X_test)
    np.save(paths["y_test"], y_test)
    train_metadata.to_csv(paths["train_window_metadata"], index=False)
    test_metadata.to_csv(paths["test_window_metadata"], index=False)
    paths["validation_report"].write_text(
        json.dumps(validation_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {name: _file_hash(path) for name, path in paths.items()}


def run_preprocessing(config: PreprocessingConfig) -> dict[str, Any]:
    warnings: list[str] = []
    train_df, test_df, rul_df = load_fd001_files(config)
    raw_checks = validate_raw_data(train_df, test_df, rul_df)
    train_with_rul, test_with_rul = add_rul_targets(train_df, test_df, rul_df)
    feature_report = build_feature_report(train_with_rul, test_with_rul)
    if feature_report["train_constant_features_detected"] != EXCLUDED_CONSTANT_FEATURES:
        warnings.append("Detected training constant features differ from approved exclusion list.")
    if feature_report["test_constant_features_detected"] != EXCLUDED_CONSTANT_FEATURES:
        warnings.append("Detected test constant features differ from approved exclusion list.")

    train_scaled, test_scaled, scaler, scaler_report = fit_and_transform(train_with_rul, test_with_rul)
    X_train, y_train, train_metadata = generate_train_windows(
        train_scaled,
        window_size=config.window_size,
        stride=config.stride,
    )
    X_test, y_test, test_metadata = generate_test_windows(
        test_scaled,
        window_size=config.window_size,
    )
    validation_checks = validate_outputs(
        config,
        train_with_rul,
        test_with_rul,
        rul_df,
        X_train,
        y_train,
        X_test,
        y_test,
        train_metadata,
        test_metadata,
        scaler_report,
    )

    validation_report: dict[str, Any] = {
        "raw_checks": raw_checks,
        "preprocessing_checks": validation_checks,
        "scaler_report": scaler_report,
        "warnings": warnings,
        "deterministic_inputs": {
            "train_sha256": _file_hash(config.train_path),
            "test_sha256": _file_hash(config.test_path),
            "rul_sha256": _file_hash(config.rul_path),
        },
    }
    artifact_hashes = save_artifacts(
        config,
        scaler,
        X_train,
        y_train,
        X_test,
        y_test,
        train_metadata,
        test_metadata,
        feature_report,
        validation_report,
    )
    validation_report["artifact_hashes"] = artifact_hashes
    (config.output_dir / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "config": config.to_dict(),
        "feature_report": feature_report,
        "validation_report": validation_report,
        "shapes": {
            "X_train": X_train.shape,
            "y_train": y_train.shape,
            "X_test": X_test.shape,
            "y_test": y_test.shape,
        },
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "selected_features": SELECTED_FEATURES,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess NASA C-MAPSS FD001 data.")
    parser.add_argument("--train-path", default="Data/train_FD001.txt")
    parser.add_argument("--test-path", default="Data/test_FD001.txt")
    parser.add_argument("--rul-path", default="Data/RUL_FD001.txt")
    parser.add_argument("--output-dir", default="artifacts/preprocessing/fd001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessingConfig(
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        rul_path=Path(args.rul_path),
        output_dir=Path(args.output_dir),
    )
    result = run_preprocessing(config)
    print("Preprocessing completed.")
    print(f"X_train shape: {result['shapes']['X_train']}")
    print(f"y_train shape: {result['shapes']['y_train']}")
    print(f"X_test shape: {result['shapes']['X_test']}")
    print(f"y_test shape: {result['shapes']['y_test']}")
    print(f"Selected features ({len(result['selected_features'])}): {result['selected_features']}")
    print(f"Training windows: {result['train_windows']}")
    print(f"Test windows: {result['test_windows']}")
    print(f"Warnings: {result['warnings'] if result['warnings'] else 'none'}")


if __name__ == "__main__":
    main()
