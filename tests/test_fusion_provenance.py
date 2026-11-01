from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest

from training.fusion_dataset import build_fusion_samples as _build_fusion_samples, embedding_digest
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

TEST_ARTIFACT_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


def build_fusion_samples(*args, **kwargs):
    kwargs.setdefault("has_log_feature", "exclude")
    return _build_fusion_samples(*args, **kwargs)


def approved(prediction_cutoff=None):
    artifact_path = str(Path(__file__).resolve())
    artifact_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_id = "synthetic-test-manifest-v1"
    sensor_embedding_hash = embedding_digest([0.0] * 64)
    text_embedding_hash = embedding_digest([0.0] * 768)
    approval = {
        "approval_register_id": "synthetic-test-register",
        "approvals": {
            field: {
                "status": "APPROVED",
                "approved_by": "fixture-test-only",
                "evidence_refs": ["synthetic-unit-test"],
            }
            for field in GENUINE_APPROVALS
        },
        "fusion_policies": {
            "missing_log": "zero_vector",
            "multiple_logs": "mean_sorted_log_id",
        },
        "prediction_cutoff": prediction_cutoff or window_cutoffs(),
        "has_log_feature": "exclude",
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_hash,
        "approved_manifest": {
            "manifest_id": manifest_id,
            "equipment_mapping_id": "synthetic-test-mapping",
            "datasets": {
                "sensor": {
                    "source_dataset": "synthetic-test-dataset",
                    "source_dataset_version": "fixture-v1",
                    "preprocessing_version": "synthetic-test-preprocess-v1",
                },
                "text": {
                    "source_dataset": "synthetic-test-log-dataset",
                    "source_dataset_version": "fixture-v1",
                    "preprocessing_version": "synthetic-test-nlp-preprocess-v1",
                },
            },
            "record_scope": {"sensor": {"scope_type": "dataset"}, "text": {"scope_type": "dataset"}},
            "artifacts": {
                "synthetic-sensor-artifact": {
                    "artifact_id": "synthetic-sensor-artifact",
                    "artifact_sha256": artifact_hash,
                    "artifact_path": artifact_path,
                    "modality": "sensor",
                    "embedding_model": "synthetic-sensor-model",
                    "embedding_revision": "fixture-v1",
                    "embedding_sha256_by_source_record_id": {"synthetic-sensor-row-17-9": sensor_embedding_hash},
                },
                "synthetic-text-artifact": {
                    "artifact_id": "synthetic-text-artifact",
                    "artifact_sha256": artifact_hash,
                    "artifact_path": artifact_path,
                    "modality": "text",
                    "embedding_model": "synthetic-text-model",
                    "embedding_revision": "fixture-v1",
                    "embedding_sha256_by_source_record_id": {
                        "source-a": text_embedding_hash,
                        "source-b": embedding_digest([2.0] * 768),
                        "source-c": embedding_digest([100.0] * 768),
                    },
                },
            },
        },
    }
    approval["approvals"]["embedding_provenance"]["manifest_id"] = manifest_id
    return approval


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
        "artifact_path": str(Path(__file__).resolve()),
        "embedding_model": "synthetic-sensor-model",
        "embedding_revision": "fixture-v1",
        "embedding_sha256": embedding_digest([0.0] * 64),
        **timestamps(),
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
        "artifact_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_path": str(Path(__file__).resolve()),
        "embedding_model": "synthetic-text-model",
        "embedding_revision": "fixture-v1",
        "embedding_sha256": embedding_digest([0.0] * 768),
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


@pytest.mark.parametrize("modality", ["sensor", "text"])
def test_genuine_caller_embedding_substitution_is_rejected(modality):
    sensor, logs = genuine_rows()
    if modality == "sensor":
        sensor[0]["sensor_embedding"] = [999.0] * 64
    else:
        logs[0]["text_embedding"] = [999.0] * 768
    with pytest.raises(ValueError, match="embedding digest|approved artifact record"):
        build_fusion_samples(
            sensor,
            logs,
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
            prediction_cutoff=window_cutoffs(),
        )
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


@pytest.mark.parametrize("field", TIMESTAMP_FIELDS)
def test_genuine_sensor_requires_each_timestamp(field):
    sensor, logs = genuine_rows()
    sensor[0].pop(field)
    with pytest.raises(ValueError, match=f"Missing or malformed {field}"):
        build_fusion_samples(
            sensor,
            [],
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
    values["event_time"] = "2025-02-01T07:00:00Z"
    values["record_creation_time"] = "2025-02-01T08:00:00Z"
    values["ingestion_time"] = "2025-02-01T09:00:00Z"
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


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        ("event_time", "record_creation_time"),
        ("record_creation_time", "ingestion_time"),
        ("ingestion_time", "availability_time"),
    ],
)
def test_each_timestamp_ordering_violation_is_rejected(earlier, later):
    values = timestamps()
    values[earlier] = "2025-02-01T11:00:00Z"
    values[later] = "2025-02-01T10:00:00Z"
    with pytest.raises(ValueError, match="Timestamp ordering violation"):
        validate_point_in_time(values, cutoff="2025-02-01T12:00:00Z", policy=time_policy())


def test_genuine_artifact_hash_mismatch_is_rejected():
    sensor, logs = genuine_rows()
    sensor[0]["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact_sha256|artifact hash"):
        build_fusion_samples(
            sensor,
            logs,
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
            prediction_cutoff=window_cutoffs(),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"runtime": {"missing_log_policy": "error"}},
        {"runtime": {"multiple_log_policy": "error"}},
        {"runtime": {"prediction_cutoff": {(17, 9): "2025-02-01T11:00:00Z"}}},
        {"runtime": {"has_log_feature": "include"}},
    ],
)
def test_runtime_policy_must_match_approval(change):
    sensor, logs = genuine_rows()
    runtime = {
        "missing_log_policy": "zero_vector",
        "multiple_log_policy": "mean_sorted_log_id",
        "prediction_cutoff": window_cutoffs(),
        "has_log_feature": "exclude",
    }
    runtime.update(change["runtime"])
    with pytest.raises(ValueError, match="do not match approval"):
        build_fusion_samples(
            sensor,
            logs,
            **runtime,
            genuine_approval=approved(),
            point_in_time_policy=time_policy(),
        )


def _temporary_manifest_case(tmp_path):
    sensor, logs = genuine_rows()
    sensor_path = tmp_path / "sensor.bin"
    text_path = tmp_path / "text.bin"
    sensor_path.write_bytes(b"approved sensor artifact")
    text_path.write_bytes(b"approved text artifact")
    sensor_hash = sha256(sensor_path.read_bytes()).hexdigest()
    text_hash = sha256(text_path.read_bytes()).hexdigest()
    sensor[0]["artifact_path"] = str(sensor_path)
    sensor[0]["artifact_sha256"] = sensor_hash
    logs[0]["artifact_path"] = str(text_path)
    logs[0]["artifact_sha256"] = text_hash
    approval = approved()
    approval["approved_manifest"]["artifacts"]["synthetic-sensor-artifact"].update(
        artifact_path=str(sensor_path), artifact_sha256=sensor_hash
    )
    approval["approved_manifest"]["artifacts"]["synthetic-text-artifact"].update(
        artifact_path=str(text_path), artifact_sha256=text_hash
    )
    return sensor, logs, approval


def test_real_temporary_artifacts_are_bound_to_the_approved_manifest(tmp_path):
    sensor, logs, approval = _temporary_manifest_case(tmp_path)
    result = build_fusion_samples(
        sensor, logs,
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
        genuine_approval=approval,
        point_in_time_policy=time_policy(),
        prediction_cutoff=window_cutoffs(),
    )
    assert result["namespace"] == "genuine"


@pytest.mark.parametrize("mutation", [
    "tamper_bytes", "wrong_digest", "artifact_id", "dataset_version",
    "equipment_mapping", "preprocessing", "different_path", "missing_file",
    "manifest_substitution", "unapproved_digest",
])
def test_manifest_bound_artifact_substitutions_fail_closed(tmp_path, mutation):
    sensor, logs, approval = _temporary_manifest_case(tmp_path)
    manifest = approval["approved_manifest"]
    if mutation == "tamper_bytes":
        Path(sensor[0]["artifact_path"]).write_bytes(b"tampered")
    elif mutation == "wrong_digest":
        sensor[0]["artifact_sha256"] = "0" * 64
    elif mutation == "artifact_id":
        sensor[0]["artifact_id"] = "not-approved"
    elif mutation == "dataset_version":
        sensor[0]["source_dataset_version"] = "other-version"
    elif mutation == "equipment_mapping":
        sensor[0]["equipment_mapping_id"] = "other-mapping"
    elif mutation == "preprocessing":
        sensor[0]["preprocessing_version"] = "other-preprocess"
    elif mutation == "different_path":
        other = tmp_path / "other.bin"
        other.write_bytes(b"approved other artifact")
        sensor[0]["artifact_path"] = str(other)
        sensor[0]["artifact_sha256"] = sha256(other.read_bytes()).hexdigest()
    elif mutation == "missing_file":
        sensor[0]["artifact_path"] = str(tmp_path / "missing.bin")
    elif mutation == "manifest_substitution":
        approval["approved_manifest"] = dict(manifest, manifest_id="substituted-manifest")
    elif mutation == "unapproved_digest":
        other = tmp_path / "other.bin"
        other.write_bytes(b"unapproved dataset artifact")
        sensor[0]["artifact_path"] = str(other)
        sensor[0]["artifact_sha256"] = sha256(other.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        build_fusion_samples(
            sensor, logs,
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
            genuine_approval=approval,
            point_in_time_policy=time_policy(),
            prediction_cutoff=window_cutoffs(),
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
            genuine_approval=approved({(17, 9): "2025-02-01T11:00:00Z"}),
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
        genuine_approval=approved({(17, 9): "2025-02-01T11:00:00Z"}),
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
    first["embedding_sha256"] = embedding_digest(first["text_embedding"])
    second = dict(logs[0], log_id="log-a", source_record_id="source-a")
    second["availability_time"] = "2025-02-01T11:00:00Z"
    second["text_embedding"] = [4.0] * 768
    second["embedding_sha256"] = embedding_digest(second["text_embedding"])
    late = dict(logs[0], log_id="log-c", source_record_id="source-c")
    late["availability_time"] = "2025-02-01T12:00:00Z"
    late["text_embedding"] = [100.0] * 768
    late["embedding_sha256"] = embedding_digest(late["text_embedding"])
    approval = approved({(17, 9): "2025-02-01T11:00:00Z"})
    approval["approved_manifest"]["artifacts"]["synthetic-text-artifact"][
        "embedding_sha256_by_source_record_id"
    ]["source-a"] = embedding_digest(second["text_embedding"])
    result = build_fusion_samples(
        sensor,
        [first, second, late],
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
        genuine_approval=approval,
        point_in_time_policy=time_policy(late_arrival_policy="exclude"),
        prediction_cutoff={(17, 9): "2025-02-01T11:00:00Z"},
    )
    assert [list(item) for item in result["log_ids_by_window"]] == [["log-a", "log-b"]]
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
