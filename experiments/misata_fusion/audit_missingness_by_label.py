from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")

sensors = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]

running = df[df["operating_mode"] != "stopped"].copy()

print("MISSINGNESS RATE BY FAILURE LABEL")
for col in sensors:
    result = running.groupby("failure_next_24h")[col].apply(
        lambda values: values.isna().mean()
    )
    print(f"\n{col}")
    print((result * 100).round(3).to_string())

print("\nMISSING SENSOR COUNT PER READING")
running["missing_sensor_count"] = running[sensors].isna().sum(axis=1)
print(pd.crosstab(
    running["missing_sensor_count"],
    running["failure_next_24h"],
    normalize="index"
).round(4).to_string())

print("\nMISSINGNESS AUDIT COMPLETE.")
