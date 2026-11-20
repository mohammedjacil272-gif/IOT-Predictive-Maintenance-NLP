from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
assets = pd.read_csv(root / "data" / "assets.csv")

df = df.merge(assets[["asset_id", "asset_type", "asset_tag"]], on="asset_id")

checks = {
    "Negative bearing temperature": df["bearing_temp_c"] < 0,
    "Zero discharge pressure": df["discharge_pressure_bar"] == 0,
    "Zero motor current": df["motor_current_a"] == 0,
    "Rotor speed above 2000": df["rotor_speed_rpm"] > 2000,
}

for label, mask in checks.items():
    subset = df[mask]
    print(f"\n{label}: {len(subset)} rows")
    if not subset.empty:
        print(pd.crosstab(
            [subset["asset_type"], subset["operating_mode"]],
            subset["asset_id"]
        ).to_string())

print("\nASSET TYPE DETAILS:")
print(assets[["asset_id", "asset_tag", "asset_type", "rated_power_kw"]].to_string(index=False))

print("\nVALUE CONTEXT AUDIT COMPLETE.")
