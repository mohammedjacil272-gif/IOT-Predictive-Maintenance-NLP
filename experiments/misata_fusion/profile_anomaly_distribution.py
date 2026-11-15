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

print("RUNNING READINGS:", len(running))

print("\n1. MISSING SENSOR VALUES BY ASSET")
print(
    running.groupby("asset_id")[sensors]
    .apply(lambda x: x.isna().sum())
    .to_string()
)

print("\n2. MISSING SENSOR VALUES BY OPERATING MODE")
print(
    running.groupby("operating_mode")[sensors]
    .apply(lambda x: x.isna().sum())
    .to_string()
)

print("\n3. SUSPICIOUS VALUES BY ASSET")
checks = {
    "negative_bearing_temp": running["bearing_temp_c"] < 0,
    "zero_discharge_pressure": running["discharge_pressure_bar"] == 0,
    "zero_motor_current": running["motor_current_a"] == 0,
    "zero_rotor_speed": running["rotor_speed_rpm"] == 0,
}

for name, mask in checks.items():
    print(f"\n{name}: {int(mask.sum())}")
    print(
        running.loc[mask]
        .groupby("asset_id")
        .size()
        .to_string()
    )

print("\n4. SUSPICIOUS VALUES BY OPERATING MODE")
for name, mask in checks.items():
    print(f"\n{name}:")
    print(
        running.loc[mask]
        .groupby("operating_mode")
        .size()
        .to_string()
    )

print("\n5. OVERALL SENSOR MISSINGNESS (%)")
print((running[sensors].isna().mean() * 100).round(3).to_string())

print("\nQUALITY PROFILE COMPLETE.")
