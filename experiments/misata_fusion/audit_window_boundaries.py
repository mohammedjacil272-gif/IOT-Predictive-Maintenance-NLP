from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

splits = {
    "TRAIN": ("2025-09-01", "2026-02-28 23:59:59"),
    "VALIDATION": ("2026-03-01", "2026-05-31 23:59:59"),
    "TEST": ("2026-06-01", "2026-08-31 23:59:59"),
}

window_hours = 30
horizon_hours = 24

print("WINDOW / HORIZON LEAKAGE AUDIT")
print("Feature history:", window_hours, "hourly readings")
print("Prediction horizon:", horizon_hours, "hours")

for name, (start, end) in splits.items():
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    part = df[df["recorded_at"].between(start, end)].copy()
    running = part[part["operating_mode"] != "stopped"]

    print(f"\n{name}")
    print("Calendar rows:", len(part))
    print("Running rows:", len(running))
    print("First timestamp:", part["recorded_at"].min())
    print("Last timestamp:", part["recorded_at"].max())

    print("Earliest feature-history start:",
          part["recorded_at"].min() - pd.Timedelta(hours=window_hours - 1))
    print("Latest prediction-horizon end:",
          part["recorded_at"].max() + pd.Timedelta(hours=horizon_hours))

print("\nBOUNDARY WINDOWS")
for boundary in [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-06-01")]:
    print(f"\nBoundary: {boundary}")
    for asset_id in sorted(df["asset_id"].unique()):
        rows = df[
            (df["asset_id"] == asset_id) &
            df["recorded_at"].between(
                boundary - pd.Timedelta(hours=30),
                boundary + pd.Timedelta(hours=24)
            )
        ]
        print(
            f"Asset {asset_id}: "
            f"{rows['recorded_at'].min()} to {rows['recorded_at'].max()} "
            f"({len(rows)} rows)"
        )

print("\nAUDIT COMPLETE — NO FILES MODIFIED.")
