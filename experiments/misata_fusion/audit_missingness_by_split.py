
from pathlib import Path
import pandas as pd

# Locate experiment directory and load original dataset
root = Path(__file__).resolve().parent
data_path = root / "data" / "sensor_readings.csv"

df = pd.read_csv(data_path)
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

df = df.sort_values(["asset_id", "recorded_at"]).reset_index(drop=True)

sensors = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]

# Chronological split boundaries
train_end = pd.Timestamp("2026-02-28 23:59:59")
val_end = pd.Timestamp("2026-05-31 23:59:59")

def assign_split(timestamp):
    if timestamp <= train_end:
        return "train"
    elif timestamp <= val_end:
        return "validation"
    return "test"

df["split"] = df["recorded_at"].apply(assign_split)

print("\nMISSINGNESS BY SPLIT AND SENSOR")
print("=" * 90)

for split in ["train", "validation", "test"]:
    part = df[df["split"] == split]

    print(f"\n{split.upper()}")
    print("-" * 90)
    print(f"Total readings: {len(part):,}")

    for sensor in sensors:
        missing = int(part[sensor].isna().sum())
        total = len(part)
        percentage = missing / total * 100 if total else 0

        print(
            f"{sensor:25s} "
            f"missing={missing:6d} "
            f"total={total:6d} "
            f"missing_pct={percentage:7.3f}%"
        )

print("\n\nMISSINGNESS BY ASSET AND SENSOR")
print("=" * 90)

for asset, group in df.groupby("asset_id"):
    print(f"\nASSET {asset}")
    print("-" * 90)

    for sensor in sensors:
        missing = int(group[sensor].isna().sum())
        total = len(group)
        percentage = missing / total * 100 if total else 0

        print(
            f"{sensor:25s} "
            f"missing={missing:6d} "
            f"total={total:6d} "
            f"missing_pct={percentage:7.3f}%"
        )

print("\n\nMISSINGNESS BY SPLIT AND ASSET")
print("=" * 90)

for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")

    part = df[df["split"] == split]

    for asset, group in part.groupby("asset_id"):
        total_values = len(group) * len(sensors)
        missing_values = int(group[sensors].isna().sum().sum())
        percentage = (
            missing_values / total_values * 100
            if total_values else 0
        )

        print(
            f"Asset {asset} | "
            f"missing_values={missing_values:5d} | "
            f"total_values={total_values:6d} | "
            f"missing_pct={percentage:7.3f}%"
        )

print("\nMISSINGNESS BY SPLIT AUDIT COMPLETE.")
