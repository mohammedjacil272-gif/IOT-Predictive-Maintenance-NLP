from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

readings = pd.read_csv(DATA / "sensor_readings.csv")
failures = pd.read_csv(DATA / "failure_events.csv")

readings["recorded_at"] = pd.to_datetime(readings["recorded_at"])
failures["failed_at"] = pd.to_datetime(failures["failed_at"])

readings["failure_next_24h"] = readings["failure_next_24h"].astype(bool)

print("\n========== LABEL ALIGNMENT AUDIT ==========")

all_results = []

for asset_id, group in readings.groupby("asset_id"):
    events = (
        failures.loc[failures["asset_id"] == asset_id, "failed_at"]
        .sort_values()
        .tolist()
    )

    group = group.sort_values("recorded_at").copy()

    next_failure = []
    for timestamp in group["recorded_at"]:
        future_events = [event for event in events if event > timestamp]
        next_failure.append(future_events[0] if future_events else pd.NaT)

    group["actual_next_failure"] = next_failure
    group["hours_to_actual_failure"] = (
        group["actual_next_failure"] - group["recorded_at"]
    ).dt.total_seconds() / 3600

    group["expected_label"] = (
        group["hours_to_actual_failure"].gt(0)
        & group["hours_to_actual_failure"].le(24)
    )

    group["label_match"] = (
        group["failure_next_24h"] == group["expected_label"]
    )

    all_results.append(group)

    positives = group[group["failure_next_24h"]]
    mismatches = group[~group["label_match"]]

    print(f"\nAsset {asset_id}")
    print("Recorded failures:", len(events))
    print("Positive labels:", len(positives))
    print("Label mismatches:", len(mismatches))

    if not positives.empty:
        print("Positive label time-to-failure range:")
        print(
            positives["hours_to_actual_failure"]
            .describe()[["min", "25%", "50%", "75%", "max"]]
            .to_string()
        )

combined = pd.concat(all_results, ignore_index=True)

print("\n========== OVERALL RESULTS ==========")
print("Total readings:", len(combined))
print("Positive labels:", int(combined["failure_next_24h"].sum()))
print("Expected positives from event timestamps:",
      int(combined["expected_label"].sum()))
print("Matching labels:", int(combined["label_match"].sum()))
print("Mismatching labels:", int((~combined["label_match"]).sum()))

print("\n========== POSITIVE LABELS WITHOUT A FUTURE EVENT ==========")
unmatched = combined[
    combined["failure_next_24h"]
    & combined["actual_next_failure"].isna()
]
print("Count:", len(unmatched))

print("\n========== INTERPRETATION ==========")
print("This compares labels against recorded failure event timestamps.")
print("It does not prove how the generator originally created labels.")
print("Inspect any mismatches before training.")
