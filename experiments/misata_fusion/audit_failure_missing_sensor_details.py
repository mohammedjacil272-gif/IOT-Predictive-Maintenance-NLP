
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent

df = pd.read_csv(root / "data" / "sensor_readings.csv")
events = pd.read_csv(root / "data" / "failure_events.csv")

df["recorded_at"] = pd.to_datetime(df["recorded_at"])
events["failed_at"] = pd.to_datetime(events["failed_at"])

sensors = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]

print("DETAILED MISSING SENSOR AUDIT")
print("=" * 70)

for _, event in events.iterrows():
    asset = event["asset_id"]
    failed_at = event["failed_at"]

    window = df[
        (df["asset_id"] == asset)
        & (df["recorded_at"] < failed_at)
        & (df["recorded_at"] >= failed_at - pd.Timedelta(hours=24))
        & (df["operating_mode"] != "stopped")
    ].copy()

    missing = window[sensors].isna().sum()

    if missing.sum() == 0:
        continue

    print(f"\nAsset {asset} | Failure: {failed_at}")

    for sensor, count in missing.items():
        if count > 0:
            timestamps = window.loc[
                window[sensor].isna(), "recorded_at"
            ].dt.strftime("%Y-%m-%d %H:%M").tolist()

            print(f"  {sensor}: {count} missing")
            print(f"    Times: {timestamps}")

print("\nDETAILED AUDIT COMPLETE.")
