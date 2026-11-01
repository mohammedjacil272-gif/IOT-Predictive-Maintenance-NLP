from datetime import datetime

import pytest

from training.fusion_dataset import build_fusion_samples
from training.fusion_provenance import (
    FINAL_EVALUATION_AUTHORIZATION,
    GENUINE_APPROVALS,
    TRAINING_AUTHORIZATION,
    TIMESTAMP_FIELDS,
    validate_named_approval,
    validate_operational_horizon,
    validate_point_in_time,
    validate_reproducibility_contract,
    validate_verification_report,
)
from training.fusion_splits import canonical_manifest

TEST_ARTIFACT_SHA256 = "0" * 64


def approved():
    return {
        "approval_register_id": "synthetic-test-register",
        "approvals": {
            field: {
                "status": "APPROVED",
                "approved_by": "fixture-test-only",
                "evidence_refs": ["synthetic-unit-test"],
            }
            for field in GENUINE_APPROVALS
        },
    }


def time_policy(**overrides):
    return {
        "approved": True,
        "timezone": "UTC",
        "timestamp_mode": "offset_aware",
        "cutoff_inclusive": True,
        "late_arrival_policy": "reject",
        **overrides,
    }


def timestamps():
    return {field: "2025-02-01T10:00:00+00:00" for field in TIMESTAMP_FIELDS}


def window_cutoffs():
    return {(17, 9): "2025-02-01T10:00:00Z"}


def genuine_rows():
    sensor = {
        "unit_id": 17,
        "cycle": 9,
        "window_start_cycle": 1,
        "window_end_cycle": 9,
        "split": "train",
        "sensor_embedding": [0.0] * 64,
        "rul": 55.0,
        "namespace": "genuine",
        "source_dataset": "synthetic-test-dataset",
        "source_dataset_version": "fixture-v1",
        "source_record_id": "synthetic-sensor-row-17-9",
        "equipment_mapping_id": "synthetic-test-mapping",
        "preprocessing_version": "synthetic-test-preprocess-v1",
        "artifact_id": "synthetic-sensor-artifact",
        "artifact_sha256": TEST_ARTIFACT_SHA256,
        "embedding_model": "synthetic-sensor-model",
        "embedding_revision": "fixture-v1",
    }
    log = {
        "log_id": "record-a",
        "source_system": "maintenance_db",
        "source_record_id": "source-a",
        "unit_id": 17,
        "cycle": 9,
        "split": "train",
        "text_embedding": [0.0] * 768,
        "namespace": "genuine",
        "source_dataset": "synthetic-test-log-dataset",
        "source_dataset_version": "fixture-v1",
        "equipment_mapping_id": "synthetic-test-mapping",
        "preprocessing_version": "synthetic-test-nlp-preprocess-v1",
        "artifact_id": "synthetic-text-artifact",
        "artifact_sha256": TEST_ARTIFACT_SHA256,
        "embedding_model": "synthetic-text-model",
        "embedding_revision": "fixture-v1",
        **timestamps(),
    }
    return [sensor], [log]


def test_genuine_records_require_all_approvals_and_timestamp_policy():
    sensor, logs = genuine_rows()
    with pytest.raises(ValueError, match="Genuine-data use is blocked"):
        build_fusion_samples(
            sensor, logs,
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
        )

    result = build_fusion_samples(
        sensor,
        logs,
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
        genuine_approval=approved(),
        point_in_time_policy=time_policy(),
        prediction_cutoff=window_cutoffs(),
    )
    assert result["namespace"] == "genuine"
    assert result["sensor_provenance_by_window"][0]["source_record_id"] == (
        "synthetic-sensor-row-17-9"
    )
    assert result["source_ids_by_window"][0][0]["source_dataset_version"] == (
        "fixture-v1"
    )
    assert result["timestamps_by_log_by_window"][0][0]["event_time"].year == 2025
    assert (
        result["raw_timestamps_by_log_by_window"][0][0]["event_time"]
        == logs[0]["event_time"]
    )
    assert result["rul"].tolist() == [55.0]


def test_genuine_records_fail_closed_without_artifact_provenance():
    sensor, logs = genuine_rows()
    del logs[0]["source_dataset_version"]
    with pytest.raises(ValueError, match="missing provenance"):
        build_fusion_samples(
            sensor, logs,
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
            prediction_cutoff=window_cutoffs(),
        )


def test_training_and_final_evaluation_authorizations_are_separate():
    register = approved()
    training_record = {
        "status": "APPROVED",
        "approved_by": "fixture-test-only",
        "evidence_refs": ["synthetic-unit-test"],
    }
    register["approvals"][TRAINING_AUTHORIZATION] = training_record
    validate_named_approval(register, TRAINING_AUTHORIZATION)
    with pytest.raises(ValueError, match=FINAL_EVALUATION_AUTHORIZATION):
        validate_named_approval(register, FINAL_EVALUATION_AUTHORIZATION)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"event_time": None}, "Missing or malformed event_time"),
        ({"event_time": "not-a-date"}, "Malformed event_time"),
        ({"event_time": "2025-02-01T10:00:00"}, "Timezone-naive event_time"),
    ],
)
def test_missing_malformed_and_naive_timestamps_are_rejected(record, message):
    values = timestamps()
    values.update(record)
    with pytest.raises(ValueError, match=message):
        validate_point_in_time(
            values,
            cutoff="2025-02-01T11:00:00Z",
            policy=time_policy(),
        )


def test_availability_after_cutoff_is_rejected_under_explicit_policy():
    values = timestamps()
    values["availability_time"] = "2025-02-01T12:00:00Z"
    with pytest.raises(ValueError, match="after prediction cutoff"):
        validate_point_in_time(
            values,
            cutoff="2025-02-01T11:00:00Z",
            policy=time_policy(late_arrival_policy="reject"),
        )


def test_unapproved_timezone_mode_and_non_boolean_cutoff_are_rejected():
    values = timestamps()
    with pytest.raises(ValueError, match="timestamp_mode"):
        validate_point_in_time(
            values,
            cutoff="2025-02-01T11:00:00Z",
            policy={key: val for key, val in time_policy().items() if key != "timestamp_mode"},
        )
    with pytest.raises(ValueError, match="cutoff_inclusive must be"):
        validate_point_in_time(
            values,
            cutoff="2025-02-01T11:00:00Z",
            policy=time_policy(cutoff_inclusive="true"),
        )


def test_availability_after_cutoff_can_be_explicitly_excluded():
    values = timestamps()
    values["availability_time"] = "2025-02-01T12:00:00Z"
    parsed = validate_point_in_time(
        values,
        cutoff=datetime.fromisoformat("2025-02-01T11:00:00+00:00"),
        policy=time_policy(late_arrival_policy="exclude"),
    )
    assert parsed["eligible"] is False


def test_availability_before_cutoff_is_eligible_and_timezone_is_normalized():
    values = timestamps()
    values["availability_time"] = "2025-02-01T15:00:00+05:30"
    parsed = validate_point_in_time(
        values,
        cutoff="2025-02-01T10:00:01Z",
        policy=time_policy(),
    )
    assert parsed["eligible"] is True
    assert parsed["availability_time"].isoformat() == "2025-02-01T09:30:00+00:00"


def test_missing_authoritative_availability_timestamp_is_rejected():
    values = timestamps()
    values.pop("availability_time")
    with pytest.raises(ValueError, match="Missing or malformed availability_time"):
        validate_point_in_time(
            values,
            cutoff="2025-02-01T11:00:00Z",
            policy=time_policy(),
        )


def test_cutoff_inclusivity_is_explicit():
    values = timestamps()
    assert validate_point_in_time(
        values, cutoff="2025-02-01T10:00:00Z",
        policy=time_policy(cutoff_inclusive=True),
    )["eligible"] is True
    assert validate_point_in_time(
        values, cutoff="2025-02-01T10:00:00Z",
        policy=time_policy(cutoff_inclusive=False, late_arrival_policy="exclude"),
    )["eligible"] is False


def test_genuine_cutoff_is_resolved_per_prediction_window():
    sensor, logs = genuine_rows()
    cutoffs = window_cutoffs()
    result = build_fusion_samples(
        sensor, logs,
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
        genuine_approval=approved(),
        point_in_time_policy=time_policy(),
        prediction_cutoff=cutoffs,
    )
    assert result["has_log"].tolist() == [True]
    with pytest.raises(ValueError, match="Missing prediction cutoff"):
        build_fusion_samples(
            sensor, logs,
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
            prediction_cutoff={},
        )


def test_equipment_split_manifest_is_deterministic_and_rejects_leakage():
    assignments = {"train": [3, 1], "validation": [2]}
    first = canonical_manifest(assignments, namespace="synthetic_fixture")
    second = canonical_manifest(
        {"validation": [2], "train": [1, 3]}, namespace="synthetic_fixture"
    )
    assert first == second
    assert first["namespace"] == "synthetic_fixture"
    assert first["final_test_composition_approval"]["status"] == "PENDING"


@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        ({"train": [1], "validation": [1]}, "Equipment leakage"),
        ({"train": [1], "validation": [2], "test": [1]}, "Equipment leakage"),
        ({"train": [1], "validation": [2], "test": [2]}, "Equipment leakage"),
    ],
)
def test_equipment_leakage_is_rejected_between_every_split_pair(assignments, message):
    with pytest.raises(ValueError, match=message):
        canonical_manifest(assignments, namespace="synthetic_fixture")


def test_split_manifest_rejects_missing_ambiguous_and_duplicate_assignments():
    with pytest.raises(ValueError, match="non-empty train and validation"):
        canonical_manifest({"train": [1]}, namespace="synthetic_fixture")
    with pytest.raises(ValueError, match=r"coverage mismatch; missing=\[3\]"):
        canonical_manifest(
            {"train": [1], "validation": [2]},
            namespace="synthetic_fixture",
            expected_equipment_ids=[1, 2, 3],
        )
    with pytest.raises(ValueError, match="Equipment leakage"):
        canonical_manifest(
            {"train": [1, 1], "validation": [2]},
            namespace="synthetic_fixture",
        )


def test_test_split_requires_explicit_evidenced_approval():
    with pytest.raises(ValueError, match="cannot be claimed approved"):
        canonical_manifest(
            {"train": [1], "validation": [2], "test": [3]},
            namespace="synthetic_fixture",
        )
    manifest = canonical_manifest(
        {"train": [1], "validation": [2], "test": [3]},
        namespace="synthetic_fixture",
        test_approval={"status": "APPROVED", "approved_by": "owner", "evidence_refs": ["ticket-1"]},
    )
    assert manifest["final_test_composition_approval"]["status"] == "APPROVED"


def test_late_log_is_excluded_but_sensor_window_is_retained():
    sensor, logs = genuine_rows()
    logs[0]["availability_time"] = "2025-02-01T12:00:00Z"
    result = build_fusion_samples(
        sensor,
        logs,
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
        genuine_approval=approved(),
        point_in_time_policy=time_policy(late_arrival_policy="exclude"),
        prediction_cutoff={(17, 9): "2025-02-01T11:00:00Z"},
    )
    assert result["has_log"].tolist() == [False]
    assert result["excluded_logs"][0]["log_id"] == "record-a"


def test_cutoff_is_applied_before_deterministic_multiple_log_aggregation():
    sensor, logs = genuine_rows()
    first = dict(logs[0], log_id="log-b", source_record_id="source-b")
    first["availability_time"] = "2025-02-01T10:30:00Z"
    first["text_embedding"] = [2.0] * 768
    second = dict(logs[0], log_id="log-a", source_record_id="source-a")
    second["availability_time"] = "2025-02-01T11:00:00Z"
    second["text_embedding"] = [4.0] * 768
    late = dict(logs[0], log_id="log-c", source_record_id="source-c")
    late["availability_time"] = "2025-02-01T12:00:00Z"
    late["text_embedding"] = [100.0] * 768
    result = build_fusion_samples(
        sensor,
        [first, second, late],
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
        genuine_approval=approved(),
        point_in_time_policy=time_policy(late_arrival_policy="exclude"),
        prediction_cutoff={(17, 9): "2025-02-01T11:00:00Z"},
    )
    assert result["log_ids_by_window"] == [["log-a", "log-b"]]
    assert torch_all_equal(result["text_embeddings"][0], 3.0)
    assert [item["log_id"] for item in result["excluded_logs"]] == ["log-c"]


def test_duplicate_source_system_ids_rejected():
    sensor, logs = genuine_rows()
    other = dict(logs[0], log_id="record-b", unit_id=17, cycle=9)
    with pytest.raises(ValueError, match="Duplicate source-system record identifier"):
        build_fusion_samples(
            sensor,
            [logs[0], other],
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
            prediction_cutoff=window_cutoffs(),
        )


def test_genuine_entity_requires_complete_matching_provenance():
    sensor, logs = genuine_rows()
    logs[0]["entities"] = [{"label": "COMPONENT"}]
    with pytest.raises(ValueError, match="Genuine entity.*missing provenance"):
        build_fusion_samples(
            sensor,
            logs,
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
            prediction_cutoff=window_cutoffs(),
        )


def test_same_source_id_across_systems_is_rejected_as_ambiguous():
    sensor, logs = genuine_rows()
    other = dict(
        logs[0],
        log_id="record-b",
        source_system="other_db",
        unit_id=17,
        cycle=9,
    )
    with pytest.raises(ValueError, match="source-system ID collision"):
        build_fusion_samples(
            sensor,
            [logs[0], other],
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
            prediction_cutoff=window_cutoffs(),
        )


def torch_all_equal(tensor, value):
    return bool((tensor == value).all().item())


def test_reproducibility_contract_records_source_lineage_without_fabrication():
    config = {
        "preprocessing_version": "preprocess-v1",
        "nlp_model_name": "model-name",
        "nlp_model_revision": "model-revision",
        "tokenizer_name": "tokenizer-name",
        "tokenizer_revision": "tokenizer-revision",
        "nlp_model_sha256": None,
        "tokenizer_sha256": None,
        "checkpoint_lineage": {
            "source_checkpoint_id": "owner-provided-id",
            "source_checkpoint_sha256": "b" * 64,
        },
    }
    validate_reproducibility_contract(config)
    with pytest.raises(ValueError, match="source_checkpoint_sha256 is required"):
        validate_reproducibility_contract(
            {
                **config,
                "checkpoint_lineage": {
                    "source_checkpoint_id": "owner-provided-id"
                },
            }
        )
    with pytest.raises(ValueError, match="SHA-256"):
        validate_reproducibility_contract({**config, "tokenizer_sha256": "invented"})


def test_operational_horizon_must_be_explicit_and_valid():
    assert validate_operational_horizon({"value": 10, "unit": "cycles"}) == {
        "value": 10.0,
        "unit": "cycles",
    }
    with pytest.raises(ValueError, match="explicitly configured"):
        validate_operational_horizon(None)
    with pytest.raises(ValueError, match="finite positive"):
        validate_operational_horizon({"value": 0, "unit": "cycles"})


def test_verification_report_must_match_exact_run_source_and_environment():
    report = {
        "git_commit": "abc123",
        "python_version": "3.13.7",
        "python_executable": "python.exe",
        "test_command": ["python", "-m", "pytest", "-q"],
        "test_status": "PASS",
        "test_passed": 51,
        "test_failed": 0,
        "checks": {
            "fusion_provenance": "PASS",
            "nlp_fd001_compatibility": "PASS",
            "baseline_protection": "PASS",
        },
    }
    validate_verification_report(
        report,
        git_commit="abc123",
        python_version="3.13.7",
        python_executable="python.exe",
    )
    with pytest.raises(ValueError, match="commit does not match"):
        validate_verification_report(
            report,
            git_commit="different",
            python_version="3.13.7",
            python_executable="python.exe",
        )
