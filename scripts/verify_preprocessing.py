"""Verify generated C-MAPSS FD001 preprocessing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.cmapss_fd001 import (
    EXPECTED_ENGINES,
    SELECTED_FEATURES,
    WINDOW_SIZE,
    PreprocessingConfig,
    run_preprocessing,
)


ARTIFACT_FILES = [
    "preprocessing_config.json",
    "feature_report.json",
    "scaler.joblib",
    "X_train.npy",
    "y_train.npy",
    "X_test.npy",
    "y_test.npy",
    "train_window_metadata.csv",
    "test_window_metadata.csv",
    "validation_report.json",
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifacts(output_dir: Path) -> dict[str, object]:
    missing = [name for name in ARTIFACT_FILES if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessing artifacts: {missing}")

    config = load_json(output_dir / "preprocessing_config.json")
    feature_report = load_json(output_dir / "feature_report.json")
    validation_report = load_json(output_dir / "validation_report.json")
    scaler = joblib.load(output_dir / "scaler.joblib")
    X_train = np.load(output_dir / "X_train.npy")
    y_train = np.load(output_dir / "y_train.npy")
    X_test = np.load(output_dir / "X_test.npy")
    y_test = np.load(output_dir / "y_test.npy")
    train_metadata = pd.read_csv(output_dir / "train_window_metadata.csv")
    test_metadata = pd.read_csv(output_dir / "test_window_metadata.csv")

    checks: dict[str, object] = {}
    checks["all_artifacts_exist"] = True
    checks["selected_features_match"] = feature_report["final_selected_features"] == SELECTED_FEATURES
    checks["selected_feature_count_is_17"] = len(feature_report["final_selected_features"]) == 17
    checks["scaler_is_standard_scaler"] = scaler.__class__.__name__ == "StandardScaler"
    checks["scaler_feature_count"] = int(scaler.n_features_in_) == len(SELECTED_FEATURES)
    checks["X_train_shape"] = list(X_train.shape)
    checks["y_train_shape"] = list(y_train.shape)
    checks["X_test_shape"] = list(X_test.shape)
    checks["y_test_shape"] = list(y_test.shape)
    checks["expected_X_test_shape"] = X_test.shape == (EXPECTED_ENGINES, WINDOW_SIZE, len(SELECTED_FEATURES))
    checks["expected_y_test_shape"] = y_test.shape == (EXPECTED_ENGINES,)
    checks["train_lengths_match"] = len(X_train) == len(y_train) == len(train_metadata)
    checks["test_lengths_match"] = len(X_test) == len(y_test) == len(test_metadata)
    checks["arrays_are_finite"] = bool(
        np.isfinite(X_train).all()
        and np.isfinite(y_train).all()
        and np.isfinite(X_test).all()
        and np.isfinite(y_test).all()
    )
    checks["test_windows_exactly_100"] = len(X_test) == EXPECTED_ENGINES
    checks["test_labels_match_metadata"] = np.array_equal(
        y_test.astype(np.float32),
        test_metadata["final_rul_from_RUL_file"].to_numpy(dtype=np.float32),
    )
    checks["window_lengths_are_30"] = bool(
        ((train_metadata["window_end_cycle"] - train_metadata["window_start_cycle"] + 1) == WINDOW_SIZE).all()
        and ((test_metadata["window_end_cycle"] - test_metadata["window_start_cycle"] + 1) == WINDOW_SIZE).all()
    )
    checks["source_split_values"] = (
        set(train_metadata["source_split"].unique()) == {"train"}
        and set(test_metadata["source_split"].unique()) == {"test"}
    )
    checks["validation_report_has_no_failed_checks"] = True
    for group_name in ("raw_checks", "preprocessing_checks"):
        for check_name, value in validation_report[group_name].items():
            if isinstance(value, bool) and not value:
                checks["validation_report_has_no_failed_checks"] = False
                checks[f"failed_{group_name}_{check_name}"] = value

    current_hashes = {name: file_hash(output_dir / name) for name in ARTIFACT_FILES}
    with tempfile.TemporaryDirectory() as temp_dir:
        rerun_output = Path(temp_dir) / "fd001"
        rerun_config = PreprocessingConfig(
            train_path=Path(config["train_path"]),
            test_path=Path(config["test_path"]),
            rul_path=Path(config["rul_path"]),
            output_dir=rerun_output,
        )
        run_preprocessing(rerun_config)
        rerun_hashes = {name: file_hash(rerun_output / name) for name in ARTIFACT_FILES}

    deterministic_names = [
        "feature_report.json",
        "X_train.npy",
        "y_train.npy",
        "X_test.npy",
        "y_test.npy",
        "train_window_metadata.csv",
        "test_window_metadata.csv",
    ]
    checks["deterministic_output"] = all(
        current_hashes[name] == rerun_hashes[name] for name in deterministic_names
    )
    checks["warnings"] = validation_report.get("warnings", [])

    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    if failed:
        raise ValueError(f"Verification failed: {failed}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FD001 preprocessing artifacts.")
    parser.add_argument("--output-dir", default="artifacts/preprocessing/fd001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = verify_artifacts(Path(args.output_dir))
    print("Preprocessing verification passed.")
    print(f"X_train shape: {checks['X_train_shape']}")
    print(f"y_train shape: {checks['y_train_shape']}")
    print(f"X_test shape: {checks['X_test_shape']}")
    print(f"y_test shape: {checks['y_test_shape']}")
    print(f"Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"Training windows: {checks['X_train_shape'][0]}")
    print(f"Test windows: {checks['X_test_shape'][0]}")
    print("Validation results:")
    for name, value in checks.items():
        if name == "warnings":
            continue
        print(f"  {name}: {value}")
    print(f"Warnings: {checks['warnings'] if checks['warnings'] else 'none'}")


if __name__ == "__main__":
    main()
