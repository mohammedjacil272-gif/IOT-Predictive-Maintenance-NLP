from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
readings = pd.read_csv(root / "data" / "sensor_readings.csv")
assets = pd.read_csv(root / "data" / "assets.csv")

merged = readings.merge(
    assets[["asset_id", "asset_type"]],
    on="asset_id",
    how="left",
    validate="many_to_one"
)

print("OPERATING MODE BY MACHINE TYPE")
print(pd.crosstab(
    merged["asset_type"],
    merged["operating_mode"],
    margins=True
).to_string())

print("\nOPERATING MODE BY ASSET")
print(pd.crosstab(
    merged["asset_id"],
    merged["operating_mode"],
    margins=True
).to_string())

print("\nASSET TYPE / MODE MAPPING COMPLETE.")
