from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from experiments.azure_pdm.anchors import read_and_audit_telemetry
from experiments.azure_pdm.dataset_validation import validate_temporal_dataset
from experiments.azure_pdm.splits import SplitPolicy
from experiments.azure_pdm.targets import FailureEvent, read_and_audit_failures, target_for_anchor
from experiments.azure_pdm.windows import LookbackWindow


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def test_lookback_is_half_open_and_elapsed_time_based():
    window = LookbackWindow(3)
    assert window.bounds(dt("2015-01-01 06:00:00")) == (
        dt("2015-01-01 03:00:00"), dt("2015-01-01 06:00:00")
    )
    assert dt("2015-01-01 06:00:00") not in window.expected_hourly_timestamps(dt("2015-01-01 06:00:00"))


def test_target_interval_excludes_anchor_and_includes_exact_horizon():
    events = (
        FailureEvent(1, dt("2015-01-02 06:00:00"), "comp1", "f1"),
        FailureEvent(1, dt("2015-01-02 06:00:00"), "comp2", "f2"),
    )
    assert target_for_anchor(events, dt("2015-01-01 06:00:00"))["target_label"] == "multi_failure"
    assert target_for_anchor(events, dt("2015-01-02 06:00:00"))["target_label"] == "none"


def test_same_component_repeats_are_not_multi_failure():
    events = (
        FailureEvent(1, dt("2015-01-02 01:00:00"), "comp1", "f1"),
        FailureEvent(1, dt("2015-01-02 02:00:00"), "comp1", "f2"),
    )
    assert target_for_anchor(events, dt("2015-01-01 06:00:00"))["target_label"] == "comp1"


def test_split_has_two_sided_purge_and_horizon_safe_rules():
    policy = SplitPolicy(dt("2015-01-01 06:00:00"), dt("2016-01-01 06:00:00"))
    assert policy.assign(dt("2015-08-08 05:00:00"))[0] is None
    assert policy.assign(dt("2015-08-09 06:00:00"))[0] == "validation"
    assert policy.assign(dt("2015-10-20 05:00:00"))[0] is None
    assert policy.assign(dt("2015-10-21 06:00:00"))[0] == "test"


def test_telemetry_audit_reports_missing_hour_and_duplicate_key(tmp_path: Path):
    path = tmp_path / "PdM_telemetry.csv"
    path.write_text(
        "datetime,machineID,volt,rotate,pressure,vibration\n"
        "2015-01-01 06:00:00,1,1,2,3,4\n"
        "2015-01-01 08:00:00,1,1,2,3,4\n"
        "2015-01-01 08:00:00,1,1,2,3,4\n",
        encoding="utf-8",
    )
    audit = read_and_audit_telemetry(path)
    assert audit.duplicate_keys == 1
    assert audit.per_machine[1]["missing_hourly_count"] == 1


def test_duplicate_failure_key_fails_closed(tmp_path: Path):
    path = tmp_path / "PdM_failures.csv"
    path.write_text(
        "datetime,machineID,failure\n"
        "2015-01-01 07:00:00,1,comp1\n"
        "2015-01-01 07:00:00,1,comp1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Failure audit failed closed"):
        read_and_audit_failures(path)


def test_generated_row_validation_rejects_future_target_event(tmp_path: Path):
    path = tmp_path / "dataset.csv"
    path.write_text(
        "sample_id,machineID,anchor_timestamp,lookback_hours,split,target_label,target_event_refs,target_event_timestamps,target_event_count\n"
        "s1,1,2015-08-09 06:00:00,3,validation,comp1,[\"f1\"],[\"2015-08-10 07:00:01\"],1\n",
        encoding="utf-8",
    )
    policy = SplitPolicy(dt("2015-01-01 06:00:00"), dt("2016-01-01 06:00:00"))
    result = validate_temporal_dataset(path, split_policy=policy)
    assert result["status"] == "FAIL"
    assert any("outside (t,t+24h]" in error for error in result["errors"])
