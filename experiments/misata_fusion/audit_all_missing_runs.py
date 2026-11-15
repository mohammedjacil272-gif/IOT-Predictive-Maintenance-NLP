
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")

df["recorded_at"] = pd.to_datetime(df["recorded_at"])
df = df.sort_values(["asset_id", "recorded_at"])

sensors = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]

print("ALL CONSECUTIVE SENSOR MISSINGNESS RUNS")
print("=" * 85)

for asset, group in df.groupby("asset_id"):
    group = group.reset_index(drop=True)

    for sensor in sensors:
        missing = group[sensor].isna()
        run_id = (~missing).cumsum()

        runs = group[missing].groupby(run_id)

        for _, run in runs:
            if len(run) >= 3:
                print(
                    f"Asset {asset} | {sensor} | "
                    f"length={len(run)} | "
                    f"start={run['recorded_at'].iloc[0]} | "
                    f"end={run['recorded_at'].iloc[-1]} | "
                    f"mode={run['operating_mode'].mode().iloc[0]}"
                )

print("\nALL MISSING-RUN AUDIT COMPLETE.")
