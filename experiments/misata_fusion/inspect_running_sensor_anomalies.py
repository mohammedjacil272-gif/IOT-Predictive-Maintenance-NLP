from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
assets = pd.read_csv(root / "data" / "assets.csv")

df = df.merge(
    assets[["asset_id", "asset_type", "asset_tag"]],
    on="asset_id"
)
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

checks = {
    "NEGATIVE BEARING TEMPERATURE": df["bearing_temp_c"] < 0,
    "RUNNING ZERO DISCHARGE PRESSURE": (
        (df["operating_mode"] != "stopped") &
        (df["discharge_pressure_bar"] == 0)
    ),
    "RUNNING ZERO MOTOR CURRENT": (
        (df["operating_mode"] != "stopped") &
        (df["motor_current_a"] == 0)
    ),
    "RUNNING ZERO ROTOR SPEED": (
        (df["operating_mode"] != "stopped") &
        (df["rotor_speed_rpm"] == 0)
    ),
}

columns = [
    "asset_id", "asset_tag", "asset_type", "recorded_at",
    "operating_mode", "vibration_mm_s", "bearing_temp_c",
    "motor_current_a", "discharge_pressure_bar",
    "rotor_speed_rpm", "failure_next_24h"
]

for label, mask in checks.items():
    subset = df[mask]
    print(f"\n{label}: {len(subset)}")
    if not subset.empty:
        print(subset[columns].to_string(index=False))

print("\nRUNNING VALUE REVIEW COMPLETE.")
