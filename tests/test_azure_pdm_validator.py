from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.azure_pdm.derived_text import (
    DERIVED_NAMESPACE,
    derive_event_text_records,
    validate_derived_synthetic_record,
)
from experiments.azure_pdm.manifest import ManifestStaleError, build_manifest, verify_manifest, write_manifest
from experiments.azure_pdm.schema import AzurePdMValidationError, TimestampPolicy, normalize_rows, read_table
from experiments.azure_pdm.temporal import FutureEventError, reject_future_events


FIXTURE_ROOT = Path(__file__).parents[1] / "experiments" / "azure_pdm" / "tests" / "fixtures"


def policy(*, allow_naive: bool = True) -> TimestampPolicy:
    return TimestampPolicy("UTC", allow_naive=allow_naive)


def test_synthetic_fixture_manifest_contains_hashes_schema_ranges_and_pass_status(tmp_path):
    manifest = build_manifest(FIXTURE_ROOT, timestamp_policy=policy())
    assert manifest["validation_status"] == "PASS"
    assert manifest["is_synthetic"] is True
    assert manifest["machine_id_coverage"]["all_event_and_telemetry_ids_present"] is True
    assert set(manifest["tables"]) == {
        "PdM_telemetry.csv", "PdM_errors.csv", "PdM_maint.csv",
        "PdM_failures.csv", "PdM_machines.csv",
    }
    for table in manifest["tables"].values():
        assert len(table["sha256"]) == 64
        assert table["row_count"] > 0
    output = tmp_path / "manifest.json"
    write_manifest(manifest, output)
    assert json.loads(output.read_text(encoding="utf-8"))["validation_status"] == "PASS"


def _copy_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for source in FIXTURE_ROOT.iterdir():
        (root / source.name).write_bytes(source.read_bytes())


def test_manifest_verification_accepts_unchanged_files_and_rejects_policy_change(tmp_path):
    dataset = tmp_path / "dataset"
    _copy_fixture(dataset)
    manifest = build_manifest(dataset, timestamp_policy=policy())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)
    verified = verify_manifest(manifest_path, dataset, timestamp_policy=policy())
    assert verified["validation_status"] == "PASS"
    with pytest.raises(ManifestStaleError, match="timestamp policy"):
        verify_manifest(manifest_path, dataset, timestamp_policy=policy(allow_naive=False))
    with pytest.raises(ManifestStaleError, match="availability"):
        verify_manifest(manifest_path, dataset, timestamp_policy=policy(), require_availability=True)


def test_manifest_verification_rejects_modified_and_missing_files(tmp_path):
    dataset = tmp_path / "dataset"
    _copy_fixture(dataset)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_manifest(dataset, timestamp_policy=policy()), manifest_path)
    telemetry = dataset / "PdM_telemetry.csv"
    telemetry.write_text(telemetry.read_text(encoding="utf-8") + "", encoding="utf-8")
    telemetry.write_text(telemetry.read_text(encoding="utf-8").replace("170.1", "170.9"), encoding="utf-8")
    with pytest.raises(ManifestStaleError, match="tables"):
        verify_manifest(manifest_path, dataset, timestamp_policy=policy())
    telemetry.write_text((FIXTURE_ROOT / "PdM_telemetry.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (dataset / "PdM_errors.csv").unlink()
    with pytest.raises(ManifestStaleError):
        verify_manifest(manifest_path, dataset, timestamp_policy=policy())


def test_manifest_writer_refuses_overwrite(tmp_path):
    output = tmp_path / "manifest.json"
    write_manifest({"sentinel": True}, output)
    with pytest.raises(FileExistsError):
        write_manifest({"sentinel": False}, output)


def test_naive_timestamps_require_explicit_allowance():
    rows, _ = read_table(FIXTURE_ROOT / "PdM_errors.csv")
    with pytest.raises(AzurePdMValidationError, match="Naive"):
        normalize_rows("PdM_errors.csv", rows, timestamp_policy=policy(allow_naive=False))


def test_aware_timestamps_are_normalized_to_utc():
    rows = [{"datetime": "2025-01-01T01:00:00+01:00", "machineID": "1", "errorID": "error1"}]
    normalized = normalize_rows("PdM_errors.csv", rows, timestamp_policy=policy(allow_naive=False))
    assert normalized[0]["event_time"].isoformat() == "2025-01-01T00:00:00+00:00"


def test_schema_rejects_wrong_columns_duplicate_rows_and_bad_values(tmp_path):
    bad = tmp_path / "PdM_machines.csv"
    bad.write_text("machineID,model,age\n1,model1,3\n1,model1,4\n", encoding="utf-8")
    for name in ("PdM_telemetry.csv", "PdM_errors.csv", "PdM_maint.csv", "PdM_failures.csv"):
        (tmp_path / name).write_text(
            ",".join({
                "PdM_telemetry.csv": ("datetime", "machineID", "volt", "rotate", "pressure", "vibration"),
                "PdM_errors.csv": ("datetime", "machineID", "errorID"),
                "PdM_maint.csv": ("datetime", "machineID", "comp"),
                "PdM_failures.csv": ("datetime", "machineID", "failure"),
            }[name]) + "\n",
            encoding="utf-8",
        )
    manifest = build_manifest(tmp_path, timestamp_policy=policy())
    assert manifest["validation_status"] == "FAIL"
    assert any("duplicate record key" in issue for issue in manifest["validation_issues"])

    bad.write_text("machineID,model,age\n1,model1,nan\n", encoding="utf-8")
    manifest = build_manifest(tmp_path, timestamp_policy=policy())
    assert manifest["validation_status"] == "FAIL"
    assert any("age must be finite" in issue for issue in manifest["validation_issues"])


def test_machine_id_coverage_is_fail_closed(tmp_path):
    for source in FIXTURE_ROOT.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    machines = tmp_path / "PdM_machines.csv"
    machines.write_text("machineID,model,age\n1,model1,3\n", encoding="utf-8")
    manifest = build_manifest(tmp_path, timestamp_policy=policy())
    assert manifest["validation_status"] == "FAIL"
    assert any("missing from PdM_machines.csv" in issue for issue in manifest["validation_issues"])


def test_event_csvs_can_supply_explicit_availability_and_policy_requires_it(tmp_path):
    dataset = tmp_path / "dataset"
    _copy_fixture(dataset)
    for filename, value in {
        "PdM_errors.csv": "2025-01-01 03:00:00",
        "PdM_maint.csv": "2025-01-01 03:00:00",
        "PdM_failures.csv": "2025-01-01 03:00:00",
    }.items():
        path = dataset / filename
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] += ",available_at"
        lines = [lines[0]] + [f"{line},{value}" for line in lines[1:]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = build_manifest(dataset, timestamp_policy=policy(), require_availability=True)
    assert manifest["validation_status"] == "PASS"
    missing_policy = build_manifest(FIXTURE_ROOT, timestamp_policy=policy(), require_availability=True)
    assert missing_policy["validation_status"] == "FAIL"
    assert any("available_at is required" in issue for issue in missing_policy["validation_issues"])


def test_owner_provided_source_ids_are_accepted_and_preserved(tmp_path):
    dataset = tmp_path / "dataset"
    _copy_fixture(dataset)
    path = dataset / "PdM_errors.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] += ",source_record_id"
    lines = [lines[0]] + [f"{line},owner-error-{index}" for index, line in enumerate(lines[1:], start=1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = build_manifest(dataset, timestamp_policy=policy())
    assert manifest["validation_status"] == "PASS"
    rows, columns = read_table(path)
    assert "source_record_id" in columns
    normalized = normalize_rows("PdM_errors.csv", rows, timestamp_policy=policy())
    assert [row["source_record_id"] for row in normalized] == ["owner-error-1", "owner-error-2"]
    assert all(row["source_record_id_kind"] == "owner_provided" for row in normalized)


def test_derived_text_is_deterministic_and_explicitly_synthetic():
    errors = [{
        "source_file": "PdM_errors.csv",
        "source_record_id": "PdM_errors.csv:2",
        "source_record_id_kind": "owner_provided",
        "event_time": policy().parse("2025-01-01 00:30:00"),
        "available_at": policy().parse("2025-01-01 00:35:00"),
        "machineID": 1,
        "errorID": "error1",
    }]
    records = derive_event_text_records(
        errors, [], prediction_cutoff="2025-01-01T01:00:00Z", timestamp_policy=policy()
    )
    assert len(records) == 1
    assert records[0]["namespace"] == DERIVED_NAMESPACE
    assert records[0]["is_synthetic"] is True
    assert records[0]["source_record_id"] == "PdM_errors.csv:2"
    assert records[0]["text"] == "structured error event error_id=error1 machine_id=1 observed_at=2025-01-01T00:30:00Z"
    validate_derived_synthetic_record(records[0])


def test_derived_text_rejects_tampering_and_genuine_namespace():
    record = {
        "source_file": "PdM_errors.csv",
        "source_record_id": "derived:PdM_errors.csv:abc",
        "source_record_id_kind": "derived_content_hash",
        "machineID": 1,
        "event_type": "error",
        "source_field": "errorID",
        "source_value": "error1",
        "event_time": "2025-01-01T00:30:00Z",
        "available_at": "2025-01-01T00:35:00Z",
        "text": "structured error event error_id=error1 machine_id=1 observed_at=2025-01-01T00:30:00Z",
        "template_version": "azure_pdm_structured_event_v1",
        "namespace": DERIVED_NAMESPACE,
        "is_synthetic": True,
    }
    validate_derived_synthetic_record(record)
    record["text"] = "invented technician symptom"
    with pytest.raises(AzurePdMValidationError, match="does not match"):
        validate_derived_synthetic_record(record)
    record["text"] = "structured error event error_id=error1 machine_id=1 observed_at=2025-01-01T00:30:00Z"
    record["namespace"] = "genuine"
    with pytest.raises(AzurePdMValidationError, match="namespace"):
        validate_derived_synthetic_record(record)


def test_future_events_are_rejected_not_silently_filtered():
    future = [{
        "source_record_id": "PdM_errors.csv:3",
        "event_time": "2025-01-01T02:00:00Z",
        "available_at": "2025-01-01T02:00:00Z",
    }]
    with pytest.raises(FutureEventError, match="exceeds cutoff"):
        reject_future_events(
            future, "2025-01-01T01:00:00Z", timestamp_policy=policy()
        )


def test_future_maintenance_and_error_events_block_derived_text():
    future = [{
        "source_file": "PdM_maint.csv",
        "source_record_id": "PdM_maint.csv:2",
        "event_time": "2025-01-01T00:30:00Z",
        "available_at": "2025-01-01T02:00:00Z",
        "machineID": 1,
        "comp": "comp1",
    }]
    with pytest.raises(FutureEventError):
        derive_event_text_records(
            [], future, prediction_cutoff="2025-01-01T01:00:00Z", timestamp_policy=policy()
        )


def test_missing_availability_is_not_inferred_from_event_time():
    record = {
        "source_file": "PdM_errors.csv",
        "source_record_id": "PdM_errors.csv:2",
        "source_record_id_kind": "owner_provided",
        "event_time": "2025-01-01T00:30:00Z",
        "machineID": 1,
        "errorID": "error1",
    }
    with pytest.raises(AzurePdMValidationError, match="Missing available_at"):
        derive_event_text_records(
            [record], [], prediction_cutoff="2025-01-01T01:00:00Z", timestamp_policy=policy()
        )


def test_fallback_source_id_is_stable_under_row_reordering_and_collisions_are_rejected():
    rows = [
        {"datetime": "2025-01-01T00:00:00Z", "machineID": "1", "errorID": "error1"},
        {"datetime": "2025-01-01T01:00:00Z", "machineID": "1", "errorID": "error2"},
    ]
    first = normalize_rows("PdM_errors.csv", rows, timestamp_policy=policy(allow_naive=False))
    second = normalize_rows("PdM_errors.csv", list(reversed(rows)), timestamp_policy=policy(allow_naive=False))
    ids_first = {row["errorID"]: row["source_record_id"] for row in first}
    ids_second = {row["errorID"]: row["source_record_id"] for row in second}
    assert ids_first == ids_second
    assert all(row["source_record_id_kind"] == "derived_content_hash" for row in first)
    duplicate_owner_ids = [
        {"datetime": "2025-01-01T00:00:00Z", "machineID": "1", "errorID": "error1", "source_record_id": "owner-1"},
        {"datetime": "2025-01-01T01:00:00Z", "machineID": "1", "errorID": "error2", "source_record_id": "owner-1"},
    ]
    with pytest.raises(AzurePdMValidationError, match="duplicate source_record_id"):
        normalize_rows("PdM_errors.csv", duplicate_owner_ids, timestamp_policy=policy(allow_naive=False))
