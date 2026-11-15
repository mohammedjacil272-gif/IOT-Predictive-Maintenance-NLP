from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
events = pd.read_csv(root / "data" / "failure_events.csv")
events["failed_at"] = pd.to_datetime(events["failed_at"])

print("FAILURE EVENTS BY MONTH:")
events["month"] = events["failed_at"].dt.to_period("M").astype(str)
print(events.groupby("month").size().to_string())

print("\nFAILURE EVENTS BY ASSET AND MONTH:")
print(pd.crosstab(events["month"], events["asset_id"]).to_string())

print("\nFIRST AND LAST FAILURE:")
print(events["failed_at"].min())
print(events["failed_at"].max())

print("\nEVENT PROFILE COMPLETE.")
