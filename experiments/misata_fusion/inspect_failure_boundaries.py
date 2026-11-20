from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
data = root / "data"

readings = pd.read_csv(data / "sensor_readings.csv")
failures = pd.read_csv(data / "failure_events.csv")

readings["recorded_at"] = pd.to_datetime(readings["recorded_at"])
failures["failed_at"] = pd.to_datetime(failures["failed_at"])
failures["restored_at"] = pd.to_datetime(failures["restored_at"])

for _, failure in failures.sort_values(["asset_id", "failed_at"]).iterrows():
    asset = failure["asset_id"]
    failed_at = failure["failed_at"]
    restored_at = failure["restored_at"]

    nearby = readings[
        (readings["asset_id"] == asset)
        & (readings["recorded_at"] >= failed_at - pd.Timedelta(hours=3))
        & (readings["recorded_at"] <= restored_at + pd.Timedelta(hours=3))
    ].copy()

    if nearby.empty:
        continue

    print(f"\nASSET {asset} | FAILURE {failed_at} | RESTORED {restored_at}")
    print(nearby[
        [
            "recorded_at",
            "operating_mode",
            "degradation_stage",
            "failure_next_24h",
            "rul_hours",
        ]
    ].to_string(index=False))

print("\nBoundary inspection complete. Original files were not modified.")
