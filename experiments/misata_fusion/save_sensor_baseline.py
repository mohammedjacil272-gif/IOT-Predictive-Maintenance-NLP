from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sensor_readings.csv"
OUT = ROOT / "processed"

EXPECTED_SOURCE_HASH = "9C7B6551E797A4CAF28AAE4399F3E1930BBA7BE0E5040E8F15A19BA5EA84AFDD"

FEATURES = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]
TARGET = "failure_next_24h"
WINDOW = 30
PURGE_HOURS = 24

SPLITS = {
    "train": ("2025-09-01 00:00:00", "2026-02-28 23:00:00"),
    "validation": ("2026-03-01 00:00:00", "2026-05-31 23:00:00"),
    "test": ("2026-06-01 00:00:00", "2026-08-31 23:00:00"),
}

EXPECTED = {
    "train": (25410, 460),
    "validation": (12957, 230),
    "test": (12979, 207),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    if not DATA.is_file():
        raise FileNotFoundError(DATA)

    if OUT.exists():
        raise FileExistsError(
            f"Refusing to write: output directory already exists: {OUT}"
        )

    source_hash_before = sha256(DATA)
    if source_hash_before.lower() != EXPECTED_SOURCE_HASH.lower():
        raise ValueError("Source CSV hash differs from the audited baseline.")

    print("Source hash verified.")
    df = pd.read_csv(DATA)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="raise")
    df = df.sort_values(["asset_id", "recorded_at"]).reset_index(drop=True)

    if df[["asset_id", "recorded_at"]].duplicated().any():
        raise ValueError("Duplicate asset/timestamp pairs found.")

    if df[TARGET].isna().any() or not df[TARGET].isin([True, False]).all():
        raise ValueError("Invalid target values.")

    split_masks = {}
    for name, (start, end) in SPLITS.items():
        split_masks[name] = df["recorded_at"].between(
            pd.Timestamp(start), pd.Timestamp(end)
        )

    total_assigned = sum(mask.astype(int) for mask in split_masks.values())
    if not (total_assigned == 1).all():
        raise ValueError("Split assignment is incomplete or overlapping.")

    train_rows = df.loc[split_masks["train"], FEATURES]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(train_rows)
    scaler.fit(train_imputed)
    transformed = scaler.transform(imputer.transform(df[FEATURES]))

    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite transformed sensor values.")

    results = {}

    for name, mask in split_masks.items():
        X, y, endpoint_times, endpoint_assets = [], [], [], []
        purged = 0
        stopped = 0
        insufficient = 0
        invalid_windows = 0

        for asset_id, group in df.groupby("asset_id", sort=True):
            global_indices = group.index.to_numpy()
            split_positions = np.flatnonzero(
                mask.loc[global_indices].to_numpy()
            )

            if len(split_positions) == 0:
                continue

            eligible_positions = split_positions[:-PURGE_HOURS]
            purged += min(PURGE_HOURS, len(split_positions))

            asset_values = transformed[global_indices]
            asset_times = group["recorded_at"].to_numpy()
            asset_modes = group["operating_mode"].to_numpy()
            asset_targets = group[TARGET].to_numpy()

            for pos in eligible_positions:
                if asset_modes[pos] == "stopped":
                    stopped += 1
                    continue

                if pos < WINDOW - 1:
                    insufficient += 1
                    continue

                start_pos = pos - WINDOW + 1
                times = pd.DatetimeIndex(asset_times[start_pos:pos + 1])

                if len(times) != WINDOW:
                    invalid_windows += 1
                    continue

                if not (
                    times.to_series().diff().dropna()
                    == pd.Timedelta(hours=1)
                ).all():
                    invalid_windows += 1
                    continue

                X.append(asset_values[start_pos:pos + 1])
                y.append(int(bool(asset_targets[pos])))
                endpoint_times.append(
                    pd.Timestamp(asset_times[pos]).isoformat()
                )
                endpoint_assets.append(asset_id)

        if not X:
            raise ValueError(f"No windows generated for {name}.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        endpoint_times = np.asarray(endpoint_times, dtype="U32")
        endpoint_assets = np.asarray(endpoint_assets)

        expected_count, expected_positive = EXPECTED[name]
        if len(X) != expected_count or int(y.sum()) != expected_positive:
            raise ValueError(
                f"{name} count mismatch: windows={len(X)}, "
                f"positives={int(y.sum())}"
            )

        if X.shape[1:] != (WINDOW, len(FEATURES)):
            raise ValueError(f"Unexpected shape for {name}: {X.shape}")

        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite values in {name} windows.")

        results[name] = {
            "X": X,
            "y": y,
            "endpoint_times": endpoint_times,
            "endpoint_assets": endpoint_assets,
        }

        print(
            f"{name}: {len(X)} windows, {int(y.sum())} positives, "
            f"{int((y == 0).sum())} negatives"
        )
        print(
            f"  Purged={purged}, stopped={stopped}, "
            f"insufficient_history={insufficient}, "
            f"invalid_windows={invalid_windows}"
        )

    source_hash_after = sha256(DATA)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source CSV changed during preprocessing.")

    metadata = {
        "dataset": "synthetic Misata industrial sensor dataset",
        "source_file": "data/sensor_readings.csv",
        "source_sha256": source_hash_before,
        "features": FEATURES,
        "target": TARGET,
        "window_hours": WINDOW,
        "purge_hours_per_asset_per_split": PURGE_HOURS,
        "split_boundaries": SPLITS,
        "imputation": "median, fitted on training-period rows only",
        "scaling": "StandardScaler, fitted on training-period rows only",
        "imputer_statistics": {
            feature: float(value)
            for feature, value in zip(FEATURES, imputer.statistics_)
        },
        "scaler_mean": {
            feature: float(value)
            for feature, value in zip(FEATURES, scaler.mean_)
        },
        "scaler_scale": {
            feature: float(value)
            for feature, value in zip(FEATURES, scaler.scale_)
        },
        "splits": {
            name: {
                "windows": int(len(values["X"])),
                "positives": int(values["y"].sum()),
                "negatives": int((values["y"] == 0).sum()),
                "X_shape": list(values["X"].shape),
                "y_shape": list(values["y"].shape),
            }
            for name, values in results.items()
        },
        "model_trained": False,
        "source_dataset_modified": False,
    }

    # Create a new output directory only after all validation succeeds.
    OUT.mkdir(parents=False, exist_ok=False)

    for name, values in results.items():
        output_file = OUT / f"{name}.npz"
        np.savez_compressed(
            output_file,
            X=values["X"],
            y=values["y"],
            endpoint_times=values["endpoint_times"],
            endpoint_assets=values["endpoint_assets"],
        )
        print("Saved:", output_file.name)

    metadata_file = OUT / "preprocessing_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("Saved:", metadata_file.name)

    if sha256(DATA) != EXPECTED_SOURCE_HASH:
        raise RuntimeError("Final source hash verification failed.")

    print("\nPASS: all three validated splits were saved.")
    print("Source CSV hash remains unchanged.")
    print("No model was trained.")
    print("No dependencies were installed.")

if __name__ == "__main__":
    main()
