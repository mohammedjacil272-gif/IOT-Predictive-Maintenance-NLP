
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

print("MISSINGNESS IN THE 24 HOURS BEFORE EACH FAILURE")
print("=" * 65)

for _, event in events.iterrows():
    asset = event["asset_id"]
    failed_at = event["failed_at"]

    window = df[
        (df["asset_id"] == asset)
        & (df["recorded_at"] < failed_at)
        & (df["recorded_at"] >= failed_at - pd.Timedelta(hours=24))
        & (df["operating_mode"] != "stopped")
    ]

    missing = int(window[sensors].isna().sum().sum())
    possible = len(window) * len(sensors)

    print(
        f"Asset {asset} | {failed_at} | "
        f"running rows={len(window)} | "
        f"missing values={missing}/{possible}"
    )

print("\nFAILURE-WINDOW MISSINGNESS AUDIT COMPLETE.")
