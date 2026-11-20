from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sensor_readings.csv"

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

def main():
    print("Loading source dataset...")
    df = pd.read_csv(DATA)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="raise")
    df = df.sort_values(["asset_id", "recorded_at"]).reset_index(drop=True)

    required = set(FEATURES + [TARGET, "asset_id", "recorded_at", "operating_mode"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["asset_id", "recorded_at"]].duplicated().any():
        raise ValueError("Duplicate asset/timestamp pairs found.")

    if df[TARGET].isna().any():
        raise ValueError("Null target labels found.")

    if not df[TARGET].isin([True, False]).all():
        raise ValueError("Target must contain only Boolean values.")

    # Assign each timestamp to exactly one chronological split.
    split_masks = {}
    for name, (start, end) in SPLITS.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        split_masks[name] = df["recorded_at"].between(start, end)

    total_assigned = sum(mask.astype(int) for mask in split_masks.values())
    if not (total_assigned == 1).all():
        raise ValueError("Some rows are missing a split or assigned to multiple splits.")

    # Fit imputation and scaling on training-period sensor readings only.
    train_rows = df.loc[split_masks["train"], FEATURES]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(train_rows)
    scaler.fit(train_imputed)

    transformed = scaler.transform(imputer.transform(df[FEATURES]))

    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite values remain after preprocessing.")

    # Check training-only fitting and transformed feature properties.
    if not np.isfinite(imputer.statistics_).all():
        raise ValueError("Invalid training imputation statistics.")

    print("\nTRAIN-ONLY PREPROCESSING")
    for feature, median in zip(FEATURES, imputer.statistics_):
        print(f"Median {feature}: {median:.6f}")
    print("Scaler fit rows:", len(train_rows))
    print("Feature count:", len(FEATURES))

    # Build windows using historical rows for each asset.
    results = {}
    for name, mask in split_masks.items():
        X, y, endpoint_times, endpoint_assets = [], [], [], []
        purged = 0
        non_running = 0
        insufficient_history = 0

        for asset_id, group in df.groupby("asset_id", sort=True):
            group = group.sort_values("recorded_at")
            global_indices = group.index.to_numpy()
            local_split_positions = np.flatnonzero(mask.loc[global_indices].to_numpy())

            if len(local_split_positions) == 0:
                continue

            # Purge the final 24 hourly endpoints for this asset and split.
            eligible_positions = local_split_positions[:-PURGE_HOURS]
            purged += min(PURGE_HOURS, len(local_split_positions))

            asset_values = transformed[global_indices]
            asset_times = group["recorded_at"].to_numpy()
            asset_modes = group["operating_mode"].to_numpy()
            asset_targets = group[TARGET].to_numpy()

            for pos in eligible_positions:
                if asset_modes[pos] == "stopped":
                    non_running += 1
                    continue

                if pos < WINDOW - 1:
                    insufficient_history += 1
                    continue

                start_pos = pos - WINDOW + 1
                times = pd.DatetimeIndex(
                    asset_times[start_pos:pos + 1]
                )

                if len(times) != WINDOW:
                    continue

                if not (times.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
                    continue

                X.append(asset_values[start_pos:pos + 1])
                y.append(bool(asset_targets[pos]))
                endpoint_times.append(pd.Timestamp(asset_times[pos]))
                endpoint_assets.append(asset_id)

        if not X:
            raise ValueError(f"No valid windows created for {name}.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.shape[1:] != (WINDOW, len(FEATURES)):
            raise ValueError(f"Unexpected window shape for {name}: {X.shape}")

        results[name] = (X, y)

        print(f"\n{name.upper()}")
        print("Windows:", len(X))
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Positive labels:", int(y.sum()))
        print("Negative labels:", int((y == 0).sum()))
        print("Purged endpoints:", purged)
        print("Stopped endpoints excluded:", non_running)
        print("Insufficient history excluded:", insufficient_history)
        print("Endpoint date range:", min(endpoint_times), "to", max(endpoint_times))

    # Verify feature count and class labels.
    assert all(X.shape[1:] == (30, 5) for X, _ in results.values())
    assert all(set(np.unique(y)).issubset({0, 1}) for _, y in results.values())

    print("\nPASS: preprocessing and window checks completed.")
    print("No model was trained.")
    print("No output dataset or model artifact was written.")
    print("Original CSV was only read.")

if __name__ == "__main__":
    main()
