from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

r = pd.read_csv(DATA / "sensor_readings.csv")
f = pd.read_csv(DATA / "failure_events.csv")

r["recorded_at"] = pd.to_datetime(r["recorded_at"])
f["failed_at"] = pd.to_datetime(f["failed_at"])

r["failure_next_24h"] = r["failure_next_24h"].astype(bool)

for asset_id, group in r.groupby("asset_id"):
    events = (
        f.loc[f.asset_id == asset_id, "failed_at"]
        .sort_values()
        .tolist()
    )

    group = group.sort_values("recorded_at").copy()

    next_events = []
    for timestamp in group.recorded_at:
        future = [event for event in events if event > timestamp]
        next_events.append(future[0] if future else pd.NaT)

    group["actual_next_failure"] = next_events
    group["hours_to_failure"] = (
        group.actual_next_failure - group.recorded_at
    ).dt.total_seconds() / 3600

    expected = (
        group.hours_to_failure.gt(0)
        & group.hours_to_failure.le(24)
    )

    mismatch = group[group.failure_next_24h != expected]

    if not mismatch.empty:
        print(f"\n========== ASSET {asset_id} ==========")
        print(mismatch[
            [
                "recorded_at",
                "failure_next_24h",
                "actual_next_failure",
                "hours_to_failure",
                "operating_mode",
                "degradation_stage",
                "rul_hours",
            ]
        ].to_string(index=False))

print("\nDiagnostic complete. Original data was not modified.")
