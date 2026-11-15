from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "data" / "sensor_readings.csv")
df["recorded_at"] = pd.to_datetime(df["recorded_at"])

df = df[df["operating_mode"] != "stopped"].copy()

splits = {
    "TRAIN": ("2025-09-01", "2026-02-28 23:00:00"),
    "VALIDATION": ("2026-03-01", "2026-05-31 23:00:00"),
    "TEST": ("2026-06-01", "2026-08-31 23:00:00"),
}

purge_hours = 24

print("TEMPORAL PURGE IMPACT")
print("Purge duration:", purge_hours, "hours")

for name, (start, end) in splits.items():
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    part = df[df["recorded_at"].between(start, end)].copy()

    if name == "TEST":
        cutoff = end - pd.Timedelta(hours=purge_hours - 1)
        eligible = part[part["recorded_at"] < cutoff]
        purged = part[part["recorded_at"] >= cutoff]
    else:
        cutoff = end - pd.Timedelta(hours=purge_hours - 1)
        eligible = part[part["recorded_at"] < cutoff]
        purged = part[part["recorded_at"] >= cutoff]

    print(f"\n{name}")
    print("Original running rows:", len(part))
    print("Eligible rows:", len(eligible))
    print("Purged rows:", len(purged))
    print("Original positives:", int(part["failure_next_24h"].sum()))
    print("Eligible positives:", int(eligible["failure_next_24h"].sum()))
    print("Purged positives:", int(purged["failure_next_24h"].sum()))
    print("Eligible positive rate (%):",
          round(eligible["failure_next_24h"].mean() * 100, 3))
    print("Purged date range:",
          purged["recorded_at"].min(), "to", purged["recorded_at"].max())

print("\nPURGE IMPACT AUDIT COMPLETE.")
