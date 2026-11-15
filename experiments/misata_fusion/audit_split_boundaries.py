from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])
events = pd.read_csv(root / "data" / "failure_events.csv")
events["failed_at"] = pd.to_datetime(events["failed_at"])

boundaries = {
    "TRAIN_VALIDATION": pd.Timestamp("2026-03-01"),
    "VALIDATION_TEST": pd.Timestamp("2026-06-01"),
}

for name, boundary in boundaries.items():
    print(f"\n{name} boundary: {boundary}")
    nearby = events[
        events["failed_at"].between(
            boundary - pd.Timedelta(hours=24),
            boundary + pd.Timedelta(hours=24)
        )
    ]
    print("Failures within +/-24 hours:", len(nearby))
    if not nearby.empty:
        print(nearby[["asset_id", "failed_at", "failure_mode"]].to_string(index=False))

    readings = df[
        df["recorded_at"].between(
            boundary - pd.Timedelta(hours=24),
            boundary + pd.Timedelta(hours=24)
        )
    ]
    print("Sensor readings in boundary window:", len(readings))

print("\nBOUNDARY AUDIT COMPLETE.")
