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

print("SENSOR BEHAVIOUR BY OPERATING MODE")

for mode, group in df.groupby("operating_mode"):
    print(f"\nMODE: {mode} | ROWS: {len(group)}")
    print(group[sensors].describe().round(3).to_string())
    print("\nMissing values:")
    print(group[sensors].isna().sum().to_string())
    print("\nZero values:")
    print(group[sensors].eq(0).sum().to_string())

print("\nSTOPPED-STATE PROFILE COMPLETE.")
