from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
events = pd.read_csv(root / "data" / "failure_events.csv")
events["failed_at"] = pd.to_datetime(events["failed_at"])

splits = {
    "TRAIN": ("2025-09-01", "2026-02-28 23:59:59"),
    "VALIDATION": ("2026-03-01", "2026-05-31 23:59:59"),
    "TEST": ("2026-06-01", "2026-08-31 23:59:59"),
}

for name, (start, end) in splits.items():
    subset = events[
        events["failed_at"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    print(f"\n{name}")
    print("Failure events:", len(subset))
    print("Assets represented:", sorted(subset["asset_id"].unique().tolist()))
    print("Events per asset:")
    print(subset.groupby("asset_id").size().to_string())

print("\nSPLIT COVERAGE CHECK COMPLETE.")
