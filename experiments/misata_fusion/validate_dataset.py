from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

FILES = {
    "assets": DATA / "assets.csv",
    "sites": DATA / "sites.csv",
    "readings": DATA / "sensor_readings.csv",
    "failures": DATA / "failure_events.csv",
    "work_orders": DATA / "work_orders.csv",
}

EXPECTED = {
    "assets": {
        "asset_id", "site_id", "asset_tag", "asset_type",
        "manufacturer", "model_code", "criticality",
        "rated_power_kw", "commissioned_at",
    },
    "sites": {"site_id", "site_name", "city", "country", "site_kind"},
    "readings": {
        "reading_id", "asset_id", "recorded_at", "operating_mode",
        "vibration_mm_s", "bearing_temp_c", "motor_current_a",
        "discharge_pressure_bar", "rotor_speed_rpm",
        "degradation_stage", "rul_hours", "failure_next_24h",
        "next_failure_mode",
    },
    "failures": {
        "failure_id", "asset_id", "failed_at", "failure_mode",
        "downtime_hours", "restored_at", "hours_since_last_repair",
    },
    "work_orders": {
        "work_order_id", "asset_id", "order_type", "opened_at",
        "closed_at", "labor_hours", "parts_cost", "priority",
    },
}

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def load_csv(name, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def main():
    section("MISATA DATASET VALIDATION")
    print("Experiment:", ROOT)
    print("Data folder:", DATA)

    frames = {
        name: load_csv(name, path)
        for name, path in FILES.items()
    }

    section("1. FILE SHAPES AND SCHEMA")
    schema_ok = True
    for name, df in frames.items():
        missing = EXPECTED[name] - set(df.columns)
        extra = set(df.columns) - EXPECTED[name]
        print(f"\n{name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print("Missing expected columns:", sorted(missing))
        print("Additional columns:", sorted(extra))
        if missing:
            schema_ok = False
    print("\nSCHEMA CHECK:", "PASS" if schema_ok else "FAIL")

    assets = frames["assets"]
    sites = frames["sites"]
    readings = frames["readings"]
    failures = frames["failures"]
    orders = frames["work_orders"]

    section("2. PRIMARY KEY CHECKS")
    primary_keys = {
        "sites": (sites, "site_id"),
        "assets": (assets, "asset_id"),
        "readings": (readings, "reading_id"),
        "failures": (failures, "failure_id"),
        "work_orders": (orders, "work_order_id"),
    }
    for name, (df, key) in primary_keys.items():
        nulls = int(df[key].isna().sum())
        duplicates = int(df[key].duplicated().sum())
        passed = nulls == 0 and duplicates == 0
        print(
            f"{name}.{key}: nulls={nulls}, duplicates={duplicates} "
            f"=> {'PASS' if passed else 'FAIL'}"
        )

    section("3. FOREIGN KEY CHECKS")
    site_ids = set(sites["site_id"].dropna())
    asset_ids = set(assets["asset_id"].dropna())

    orphan_assets = int((~assets["site_id"].isin(site_ids)).sum())
    print("Assets with unknown site_id:", orphan_assets)

    for name, df in [
        ("readings", readings),
        ("failures", failures),
        ("work_orders", orders),
    ]:
        orphan_count = int((~df["asset_id"].isin(asset_ids)).sum())
        print(f"{name} with unknown asset_id: {orphan_count}")

    section("4. TIMESTAMP AND READING CONTINUITY")
    readings["recorded_at"] = pd.to_datetime(
        readings["recorded_at"], errors="coerce"
    )
    failures["failed_at"] = pd.to_datetime(
        failures["failed_at"], errors="coerce"
    )
    failures["restored_at"] = pd.to_datetime(
        failures["restored_at"], errors="coerce"
    )
    orders["opened_at"] = pd.to_datetime(
        orders["opened_at"], errors="coerce"
    )
    orders["closed_at"] = pd.to_datetime(
        orders["closed_at"], errors="coerce"
    )

    print("Invalid reading timestamps:", int(readings["recorded_at"].isna().sum()))
    print("Invalid failure timestamps:", int(failures["failed_at"].isna().sum()))
    print("Invalid work-order opening timestamps:", int(orders["opened_at"].isna().sum()))

    duplicate_time_keys = int(
        readings.duplicated(["asset_id", "recorded_at"]).sum()
    )
    print("Duplicate asset/timestamp pairs:", duplicate_time_keys)

    for asset_id, group in readings.groupby("asset_id"):
        times = group["recorded_at"].dropna().sort_values()
        if times.empty:
            print(f"{asset_id}: no valid timestamps")
            continue
        diffs = times.diff().dropna()
        gaps = int((diffs > pd.Timedelta(hours=1)).sum())
        irregular = int((diffs != pd.Timedelta(hours=1)).sum())
        print(
            f"{asset_id}: rows={len(group)}, "
            f"range={times.min()} to {times.max()}, "
            f"gaps_over_1h={gaps}, non_hourly_intervals={irregular}"
        )

    section("5. MISSING SENSOR VALUES")
    sensor_columns = [
        "vibration_mm_s",
        "bearing_temp_c",
        "motor_current_a",
        "discharge_pressure_bar",
        "rotor_speed_rpm",
    ]
    for column in sensor_columns:
        count = int(readings[column].isna().sum())
        percentage = 100 * count / len(readings) if len(readings) else 0
        print(f"{column}: {count:,} missing ({percentage:.2f}%)")

    print("\nOther columns:")
    for column in [
        "degradation_stage", "rul_hours",
        "failure_next_24h", "next_failure_mode",
    ]:
        count = int(readings[column].isna().sum())
        print(f"{column}: {count:,} missing")

    section("6. TARGET LABEL DISTRIBUTION")
    target = readings["failure_next_24h"]
    print("Raw target values:")
    print(target.value_counts(dropna=False).to_string())
    print("\nTarget proportions:")
    print(target.value_counts(normalize=True, dropna=False).to_string())

    valid_labels = set(target.dropna().unique())
    print("\nObserved non-null label values:", valid_labels)
    print("Null target labels:", int(target.isna().sum()))

    section("7. FAILURE AND WORK-ORDER TIMING")
    print("Failures per asset:")
    print(failures.groupby("asset_id").size().to_string())

    print("\nWork orders per asset:")
    print(orders.groupby("asset_id").size().to_string())

    print("\nWork-order types:")
    print(orders["order_type"].value_counts(dropna=False).to_string())

    print("\nWork-order priorities:")
    print(orders["priority"].value_counts(dropna=False).to_string())

    # Timing association only; this does not make work orders valid model features.
    timing_rows = []
    for _, failure in failures.dropna(
        subset=["asset_id", "failed_at"]
    ).iterrows():
        same_asset = orders[orders["asset_id"] == failure["asset_id"]]
        deltas = (
            same_asset["opened_at"] - failure["failed_at"]
        ).dropna().dt.total_seconds() / 3600

        nearby = deltas[(deltas >= -24) & (deltas <= 72)]
        if not nearby.empty:
            timing_rows.append({
                "asset_id": failure["asset_id"],
                "failure_id": failure["failure_id"],
                "nearby_orders": len(nearby),
                "orders_before_failure": int((nearby < 0).sum()),
                "orders_after_failure": int((nearby > 0).sum()),
            })

    print(
        "\nFailures with a work order opened within "
        "24h before to 72h after:",
        len(timing_rows),
        "of",
        len(failures),
    )
    if timing_rows:
        timing = pd.DataFrame(timing_rows)
        print("Failures with only post-failure nearby orders:",
              int(((timing["orders_before_failure"] == 0) &
                   (timing["orders_after_failure"] > 0)).sum()))

    section("8. MODEL INPUT LEAKAGE GUARD")
    prohibited = [
        "degradation_stage",
        "rul_hours",
        "next_failure_mode",
        "failure_next_24h",
    ]
    print("Do NOT use these as baseline input features:")
    for column in prohibited:
        print("-", column)

    print("\nFailure event fields and work-order fields must also be")
    print("excluded from sensor inputs unless their availability before")
    print("each prediction timestamp is independently established.")

    section("9. SUMMARY")
    print("Schema check:", "PASS" if schema_ok else "FAIL")
    print("Rows:", len(readings))
    print("Assets:", readings["asset_id"].nunique())
    print("Failures:", len(failures))
    print("Work orders:", len(orders))
    print("\nThis is a structural/profile validation, not model validation.")
    print("Dataset is synthetic. Do not claim industrial validation.")
    print("Fusion status must remain: NOT_VALIDATED")

if __name__ == "__main__":
    main()
