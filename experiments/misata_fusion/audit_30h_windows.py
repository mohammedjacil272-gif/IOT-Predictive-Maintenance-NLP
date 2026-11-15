from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

splits = {
    "TRAIN": ("2025-09-01", "2026-02-27 23:00:00"),
    "VALIDATION": ("2026-03-01", "2026-05-30 23:00:00"),
    "TEST": ("2026-06-01", "2026-08-30 23:00:00"),
}

window_hours = 30

print("30-HOUR HISTORICAL WINDOW AUDIT")

for name, (start, end) in splits.items():
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total_endpoints = 0
    valid_windows = 0
    windows_with_stopped = 0
    windows_with_missing_sensors = 0
    missing_window_values = 0

    for asset_id, asset in df.groupby("asset_id"):
        asset = asset.sort_values("recorded_at").reset_index(drop=True)

        stopped = asset["operating_mode"].eq("stopped").astype(int)
        stopped_in_window = stopped.rolling(window_hours).sum()

        sensor_columns = [
            "vibration_mm_s", "bearing_temp_c", "motor_current_a",
            "discharge_pressure_bar", "rotor_speed_rpm"
        ]
        missing_per_row = asset[sensor_columns].isna().sum(axis=1)
        missing_in_window = missing_per_row.rolling(window_hours).sum()

        for i in range(window_hours - 1, len(asset)):
            timestamp = asset.loc[i, "recorded_at"]

            if not (start <= timestamp <= end):
                continue
            if asset.loc[i, "operating_mode"] == "stopped":
                continue

            total_endpoints += 1

            history = asset.loc[i - window_hours + 1:i, "recorded_at"]
            expected = pd.date_range(
                end=timestamp, periods=window_hours, freq="h"
            )

            if history.reset_index(drop=True).equals(
                pd.Series(expected)
            ):
                valid_windows += 1

            if stopped_in_window.iloc[i] > 0:
                windows_with_stopped += 1

            missing_count = int(missing_in_window.iloc[i])
            if missing_count > 0:
                windows_with_missing_sensors += 1
                missing_window_values += missing_count

    print(f"\n{name}")
    print("Eligible running endpoints:", total_endpoints)
    print("Valid consecutive 30-hour windows:", valid_windows)
    print("Windows containing stopped readings:", windows_with_stopped)
    print("Windows containing sensor missingness:", windows_with_missing_sensors)
    print("Total missing sensor values across windows:", missing_window_values)

print("\nWINDOW AUDIT COMPLETE — NO FILES MODIFIED.")
