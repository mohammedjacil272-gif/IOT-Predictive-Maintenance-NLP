from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
events = pd.read_csv(root / "data" / "failure_events.csv")

df["recorded_at"] = pd.to_datetime(df["recorded_at"])
events["failed_at"] = pd.to_datetime(events["failed_at"])

for _, event in events.iterrows():
    asset = event["asset_id"]
    failure = event["failed_at"]

    window = df[
        (df["asset_id"] == asset) &
        (df["recorded_at"] >= failure - pd.Timedelta(hours=48)) &
        (df["recorded_at"] <= failure)
    ].copy()

    if window.empty:
        continue

    print(
        f"\nASSET {asset} | FAILURE {failure} | "
        f"MODE COUNTS: {window['operating_mode'].value_counts().to_dict()}"
    )

    print(window[
        ["recorded_at", "operating_mode", "failure_next_24h"]
    ].tail(8).to_string(index=False))

print("\nFAILURE-RELATED OPERATING MODE AUDIT COMPLETE.")
