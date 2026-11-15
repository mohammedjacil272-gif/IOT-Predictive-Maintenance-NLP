
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")

sensors = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]

print("\nMISSINGNESS BY OPERATING MODE")
print("=" * 100)

for mode, group in df.groupby("operating_mode", dropna=False):
    print(f"\nMODE: {mode}")
    print("-" * 100)
    print(f"Total readings: {len(group):,}")

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

print("\n\nMISSING SENSOR COUNT PER READING")
print("=" * 100)

df["missing_sensor_count"] = df[sensors].isna().sum(axis=1)

print(
    df.groupby("operating_mode")["missing_sensor_count"]
    .agg(["count", "mean", "median", "max"])
    .to_string()
)

print("\n\nMISSINGNESS BY MODE AND FAILURE LABEL")
print("=" * 100)

for mode, group in df.groupby("operating_mode"):
    print(f"\nMODE: {mode}")

    print(
        group.groupby("failure_next_24h")["missing_sensor_count"]
        .agg(["count", "mean", "median", "max"])
        .to_string()
    )

print("\nMISSINGNESS BY OPERATING MODE AUDIT COMPLETE.")
