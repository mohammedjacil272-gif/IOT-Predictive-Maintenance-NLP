from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])
df = df.sort_values(["asset_id", "recorded_at"])

df["previous_mode"] = df.groupby("asset_id")["operating_mode"].shift(1)
df["mode_changed"] = (
    df["operating_mode"] != df["previous_mode"]
) & df["previous_mode"].notna()

running = df[df["operating_mode"] != "stopped"].copy()

print("MODE TRANSITION RATE")
print(
    running.groupby("failure_next_24h")["mode_changed"]
    .agg(["count", "sum", "mean"])
    .to_string()
)

print("\nTRANSITIONS BY CURRENT MODE")
print(pd.crosstab(
    running["operating_mode"],
    running["mode_changed"],
    normalize="index"
).round(4).to_string())

print("\nMODE TRANSITION AUDIT COMPLETE.")
