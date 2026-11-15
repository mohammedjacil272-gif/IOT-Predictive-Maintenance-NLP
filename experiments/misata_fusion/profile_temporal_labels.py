from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

df = df[df["operating_mode"] != "stopped"].copy()
df["failure_next_24h"] = df["failure_next_24h"].astype(bool)

print("RUNNING DATA DATE RANGE:")
print(df["recorded_at"].min(), "to", df["recorded_at"].max())

print("\nMONTHLY LABEL DISTRIBUTION:")
df["month"] = df["recorded_at"].dt.to_period("M").astype(str)
print(pd.crosstab(df["month"], df["failure_next_24h"]))

print("\nPER-ASSET MONTHLY POSITIVES:")
print(
    df[df["failure_next_24h"]]
    .pivot_table(
        index="month",
        columns="asset_id",
        values="reading_id",
        aggfunc="count",
        fill_value=0
    )
)

print("\nTEMPORAL PROFILE COMPLETE.")
