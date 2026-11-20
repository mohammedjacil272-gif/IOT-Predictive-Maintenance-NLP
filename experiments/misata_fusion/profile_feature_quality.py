from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

features = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_bar",
    "rotor_speed_rpm",
]

df = df[df["operating_mode"] != "stopped"].copy()

df["partition"] = "EXCLUDED"
df.loc[df["recorded_at"] < "2026-03-01", "partition"] = "TRAIN"
df.loc[
    (df["recorded_at"] >= "2026-03-01") &
    (df["recorded_at"] < "2026-06-01"), "partition"
] = "VALIDATION"
df.loc[df["recorded_at"] >= "2026-06-01", "partition"] = "TEST"

for name in ["TRAIN", "VALIDATION", "TEST"]:
    part = df[df["partition"] == name]
    print(f"\n{name} FEATURE PROFILE")
    print(part[features].describe().round(3).to_string())
    print("\nMissing values:")
    print(part[features].isna().sum().to_string())

print("\nFEATURE PROFILE COMPLETE.")
