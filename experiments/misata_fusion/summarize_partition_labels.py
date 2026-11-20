from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
events = pd.read_csv(root / "data" / "failure_events.csv")

df["recorded_at"] = pd.to_datetime(df["recorded_at"])
events["failed_at"] = pd.to_datetime(events["failed_at"])

df = df[df["operating_mode"] != "stopped"].copy()
df["failure_next_24h"] = df["failure_next_24h"].astype(bool)

splits = [
    ("TRAIN", "2025-09-01", "2026-02-28 23:59:59"),
    ("VALIDATION", "2026-03-01", "2026-05-31 23:59:59"),
    ("TEST", "2026-06-01", "2026-08-31 23:59:59"),
]

for name, start, end in splits:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    subset = df[df["recorded_at"].between(start, end)]
    actual_events = events[events["failed_at"].between(start, end)]

    print(f"\n{name}")
    print("Period:", start, "to", end)
    print("Running readings:", len(subset))
    print("Positive labels:", int(subset["failure_next_24h"].sum()))
    print("Negative labels:", int((~subset["failure_next_24h"]).sum()))
    print("Actual failure events:", len(actual_events))
    print("Positive rate (%):", round(subset["failure_next_24h"].mean() * 100, 3))

print("\nPARTITION LABEL SUMMARY COMPLETE.")
