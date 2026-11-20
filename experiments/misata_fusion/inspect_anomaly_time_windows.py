from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

cases = [
    (4, "2025-12-12 22:00:00"),
    (4, "2026-01-06 06:00:00"),
    (4, "2026-07-08 23:00:00"),
    (1, "2026-01-08 00:00:00"),
    (1, "2025-09-08 21:00:00"),
    (3, "2025-09-19 03:00:00"),
]

columns = [
    "recorded_at", "operating_mode", "vibration_mm_s",
    "bearing_temp_c", "motor_current_a",
    "discharge_pressure_bar", "rotor_speed_rpm",
    "failure_next_24h"
]

for asset_id, timestamp in cases:
    target = pd.Timestamp(timestamp)
    subset = df[
        (df["asset_id"] == asset_id) &
        (df["recorded_at"].between(
            target - pd.Timedelta(hours=3),
            target + pd.Timedelta(hours=3)
        ))
    ]

    print(f"\nASSET {asset_id} — AROUND {target}")
    print(subset[columns].to_string(index=False))

print("\nTEMPORAL ANOMALY REVIEW COMPLETE.")
