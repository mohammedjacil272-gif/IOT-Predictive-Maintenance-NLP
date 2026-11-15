
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")

df["recorded_at"] = pd.to_datetime(df["recorded_at"])

sensors = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]

# Chronological splits
train_end = pd.Timestamp("2026-02-28 23:59:59")
val_end = pd.Timestamp("2026-05-31 23:59:59")

df["split"] = "test"
df.loc[df["recorded_at"] <= val_end, "split"] = "validation"
df.loc[df["recorded_at"] <= train_end, "split"] = "train"

# Restrict analysis to running readings
running = df[df["operating_mode"] != "stopped"].copy()

running["missing_sensor_count"] = (
    running[sensors].isna().sum(axis=1)
)

print("\nMISSINGNESS BY SPLIT, ASSET AND FAILURE LABEL")
print("=" * 105)

for split in ["train", "validation", "test"]:
    print(f"\n{'=' * 30} {split.upper()} {'=' * 30}")

    part = running[running["split"] == split]

    for asset, group in part.groupby("asset_id"):
        print(f"\nASSET {asset}")

        summary = (
            group.groupby("failure_next_24h")[
                "missing_sensor_count"
            ]
            .agg(["count", "mean", "median", "max"])
        )

        print(summary.to_string())

print("\n\nSENSOR-SPECIFIC MISSINGNESS BY FAILURE LABEL")
print("=" * 105)

for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")

    part = running[running["split"] == split]

    for sensor in sensors:
        print(f"\nSensor: {sensor}")

        summary = (
            part.groupby("failure_next_24h")[sensor]
            .apply(lambda values: values.isna().mean() * 100)
        )

        print(summary.to_string())

print("\nMISSINGNESS-LABEL AUDIT COMPLETE.")
