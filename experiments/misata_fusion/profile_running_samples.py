from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")

print("ALL READINGS:", len(df))
print("\nOPERATING MODE COUNTS:")
print(df["operating_mode"].value_counts(dropna=False))

running = df[df["operating_mode"] != "stopped"].copy()

print("\nRUNNING READINGS:", len(running))
print("\nRUNNING TARGET DISTRIBUTION:")
print(running["failure_next_24h"].value_counts(dropna=False))
print("\nRUNNING POSITIVE RATE (%):",
      round(running["failure_next_24h"].astype(bool).mean() * 100, 3))

print("\nRUNNING ROWS BY ASSET:")
print(running.groupby("asset_id").size())

print("\nRUNNING POSITIVES BY ASSET:")
print(running[running["failure_next_24h"].astype(bool)].groupby("asset_id").size())

print("\nREAD-ONLY PROFILE COMPLETE.")
