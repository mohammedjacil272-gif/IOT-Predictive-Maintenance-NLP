"""Synthetic end-to-end tests for the NLP adapter and fusion loader.

All records in this module are generated fixtures. The timestamp-contract
fixtures use the loader's provenance-gated code path only to exercise cutoff
validation; they are not genuine source records or evidence of genuine-data
compatibility.
"""

import hashlib
import tempfile

import pytest
import torch

from training.fusion_dataset import (
    UnmatchedTextRecordsError,
    build_fusion_samples,
    embedding_digest,
)
from training.fusion_provenance import GENUINE_APPROVALS, TIMESTAMP_FIELDS
from training.nlp_fusion_adapter import adapt_nlp_embedding_metadata


def _metadata(index, log_id, unit_id, cycle, **extra):
    return {
        "row_index": index,
        "log_id": log_id,
        "unit_id": str(unit_id),
        "cycle": str(cycle),
        "source_split": "fixture",
        "model_name": "fixture-bert",
        "max_length": "96",
        "pooling": "mean",
        "embedding_dim": "768",
        **extra,
    }


def _adapt(rows, embeddings, *, processed=None):
    report = {
        "dataset_type": "synthetic_fixture",
        "number_of_logs": len(rows),
        "embedding_shape": list(embeddings.shape),
        "embedding_finite": bool(torch.isfinite(embeddings).all().item()),
        "official_test_evaluation_performed": False,
        "model_name": rows[0]["model_name"],
        "max_length": int(rows[0]["max_length"]),
        "pooling": rows[0]["pooling"],
    }
    processed_logs = processed or [
        {field: row[field] for field in ("log_id", "unit_id", "cycle", "source_split")}
        for row in rows
    ]
    return adapt_nlp_embedding_metadata(
        rows,
        embeddings,
        processed_logs=processed_logs,
        verification_report=report,
    )


def _sensor(unit_id, cycle, *, split="fixture", value=1.0):
    return {
        "unit_id": unit_id,
        "cycle": cycle,
        "split": split,
        "sensor_embedding": [value] * 64,
        "rul": float(cycle),
        "namespace": "synthetic_fixture",
    }


def _fuse(sensors, texts, *, missing="zero_vector", multiple="mean_sorted_log_id"):
    # These are explicit test choices, not approved project policies.
    return build_fusion_samples(
        sensors,
        texts,
        missing_log_policy=missing,
        multiple_log_policy=multiple,
    )


def test_basic_adapter_to_loader_integration_aligns_by_unit_cycle_and_dimensions():
    rows = [_metadata(0, "synthetic-log-b", 8, 12), _metadata(1, "synthetic-log-a", 3, 7)]
    vectors = torch.stack([torch.full((768,), 2.0), torch.full((768,), 5.0)])
    records = _adapt(rows, vectors)

    result = _fuse([_sensor(3, 7), _sensor(8, 12)], records)

    assert result["namespace"] == "synthetic_fixture"
    assert list(zip(result["unit_id"].tolist(), result["cycle"].tolist())) == [(3, 7), (8, 12)]
    assert result["sensor_embeddings"].shape == (2, 64)
    assert result["text_embeddings"].shape == (2, 768)
    assert result["has_log"].tolist() == [True, True]
    assert result["log_ids_by_window"] == [["synthetic-log-a"], ["synthetic-log-b"]]
    assert result["text_embeddings"][:, 0].tolist() == [5.0, 2.0]
    assert result["split"] == ("fixture", "fixture")


def test_multiple_logs_mean_is_sorted_deterministic_and_keeps_each_log_provenance():
    rows = [
        _metadata(0, "synthetic-log-z", 4, 9, source_record_id="synthetic-source-z",
                 entities=[{"log_id": "synthetic-log-z", "entity": "pump-z"}], severity_score=0.8),
        _metadata(1, "synthetic-log-a", 4, 9, source_record_id="synthetic-source-a",
                 entities=[{"log_id": "synthetic-log-a", "entity": "pump-a"}], severity_score=0.2),
    ]
    records = _adapt(rows, torch.stack([torch.full((768,), 8.0), torch.full((768,), 2.0)]))

    first = _fuse([_sensor(4, 9)], records, multiple="mean_sorted_log_id")
    second = _fuse([_sensor(4, 9)], list(reversed(records)), multiple="mean_sorted_log_id")

    assert first["text_embeddings"][0, 0].item() == 5.0
    assert torch.equal(first["text_embeddings"], second["text_embeddings"])
    assert first["log_ids_by_window"] == [["synthetic-log-a", "synthetic-log-z"]]
    assert [entry["source_record_id"] for entry in first["source_ids_by_window"][0]] == [
        "synthetic-source-a", "synthetic-source-z"
    ]
    assert [entry["severity_score"] for entry in first["severity_by_log_by_window"][0]] == [0.2, 0.8]
    assert [entity["entity"] for entity in first["entities_by_window"][0]] == ["pump-a", "pump-z"]

    with pytest.raises(ValueError, match="Multiple technician logs"):
        _fuse([_sensor(4, 9)], records, multiple="error")


def test_missing_log_policies_are_explicit_and_zero_vector_preserves_window():
    rows = [_metadata(0, "synthetic-present", 5, 2)]
    records = _adapt(rows, torch.full((1, 768), 3.0))
    sensors = [_sensor(5, 2), _sensor(5, 3)]

    result = _fuse(sensors, records, missing="zero_vector")

    assert result["unit_id"].tolist() == [5, 5]
    assert result["cycle"].tolist() == [2, 3]
    assert result["sensor_embeddings"].shape == (2, 64)
    assert result["text_embeddings"].shape == (2, 768)
    assert result["has_log"].tolist() == [True, False]
    assert torch.count_nonzero(result["text_embeddings"][1]).item() == 0
    assert result["log_ids_by_window"] == [["synthetic-present"], []]

    with pytest.raises(ValueError, match="Missing technician log"):
        _fuse(sensors, records, missing="error")


def _synthetic_timestamp_records(log_specs):
    """Create non-real contract fixtures for point-in-time gate tests."""
    # Create a real temporary artifact so provenance validation can verify the
    # digest against actual bytes instead of the test module itself.
    artifact_bytes = b"synthetic test artifact bytes; not real project data"
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_file = tempfile.NamedTemporaryFile(
        prefix="synthetic_fusion_artifact_", suffix=".bin", delete=False
    )
    artifact_file.write(artifact_bytes)
    artifact_file.close()
    synthetic_artifact_path = artifact_file.name

    sensor_vector = [0.5] * 64
    sensor_source_id = "synthetic-sensor-21-6"
    sensor_artifact_id = "synthetic-sensor-artifact"
    text_artifact_id = "synthetic-text-artifact"
    manifest_id = "synthetic-test-manifest-v1"

    sensor = {
        "unit_id": 21,
        "cycle": 6,
        "event_time": "2025-01-01T11:00:00Z",
        "window_start_cycle": 1,
        "window_end_cycle": 6,
        "split": "train",
        "sensor_embedding": sensor_vector,
        "embedding_sha256": embedding_digest(sensor_vector),
        "rul": 10.0,
        "namespace": "genuine",
        "source_dataset": "synthetic-test-only",
        "source_dataset_version": "fixture",
        "source_record_id": sensor_source_id,
        "equipment_mapping_id": "synthetic-test-mapping",
        "preprocessing_version": "fixture",
        "artifact_id": sensor_artifact_id,
        "artifact_sha256": digest,
        "embedding_model": "synthetic-model",
        "embedding_revision": "fixture",
        "artifact_path": synthetic_artifact_path,
    }

    # Populate every required point-in-time field on the sensor fixture too.
    # Text records already receive the same timestamp fields below.
    sensor.update({field: "2025-01-01T10:00:00Z" for field in TIMESTAMP_FIELDS})

    logs = []
    text_hashes = {}

    for log_id, source_id, availability, vector in log_specs:
        # Non-finite vectors must reach the loader's validation path. A fixture
        # placeholder digest avoids JSON hashing NaN before the loader can reject it.
        vector_digest = (
            embedding_digest(vector)
            if torch.isfinite(torch.as_tensor(vector, dtype=torch.float32)).all().item()
            else digest
        )
        log = {
            "log_id": log_id,
            "source_system": "synthetic-test-system",
            "source_record_id": source_id,
            "unit_id": 21,
            "cycle": 6,
            "split": "train",
            "text_embedding": vector,
            "embedding_sha256": vector_digest,
            "namespace": "genuine",
            "source_dataset": "synthetic-test-only",
            "source_dataset_version": "fixture",
            "equipment_mapping_id": "synthetic-test-mapping",
            "preprocessing_version": "fixture",
            "artifact_id": text_artifact_id,
            "artifact_sha256": digest,
            "embedding_model": "synthetic-model",
            "embedding_revision": "fixture",
            "artifact_path": synthetic_artifact_path,
        }

        log.update({field: "2025-01-01T10:00:00Z" for field in TIMESTAMP_FIELDS})
        log["availability_time"] = availability
        logs.append(log)
        text_hashes[source_id] = vector_digest

    manifest = {
        "manifest_id": manifest_id,
        "equipment_mapping_id": "synthetic-test-mapping",
        "datasets": {
            "sensor": {
                "source_dataset": "synthetic-test-only",
                "source_dataset_version": "fixture",
                "preprocessing_version": "fixture",
            },
            "text": {
                "source_dataset": "synthetic-test-only",
                "source_dataset_version": "fixture",
                "preprocessing_version": "fixture",
            },
        },
        "record_scope": {
            "sensor": {"scope_type": "dataset"},
            "text": {"scope_type": "dataset"},
        },
        "artifacts": {
            sensor_artifact_id: {
                "artifact_id": sensor_artifact_id,
                "artifact_sha256": digest,
                "artifact_path": synthetic_artifact_path,
                "modality": "sensor",
                "embedding_model": "synthetic-model",
                "embedding_revision": "fixture",
                "embedding_sha256_by_source_record_id": {
                    sensor_source_id: embedding_digest(sensor_vector),
                },
            },
            text_artifact_id: {
                "artifact_id": text_artifact_id,
                "artifact_sha256": digest,
                "artifact_path": synthetic_artifact_path,
                "modality": "text",
                "embedding_model": "synthetic-model",
                "embedding_revision": "fixture",
                "embedding_sha256_by_source_record_id": text_hashes,
            },
        },
    }

    approvals = {
        "approval_register_id": "synthetic-test-only-register",
        "approved_manifest": manifest,
        # These fixture approval fields must match the runtime policy arguments
        # passed by _cutoff_fuse; they are synthetic and not real approvals.
        "fusion_policies": {
            "missing_log": "zero_vector",
            "multiple_logs": "mean_sorted_log_id",
        },
        "prediction_cutoff": {(21, 6): "2025-01-01T11:00:00Z"},
        "has_log_feature": "include",
        "approvals": {
            field: {
                "status": "APPROVED",
                "approved_by": "synthetic-test-only",
                "evidence_refs": ["synthetic-test-fixture"],
            }
            for field in GENUINE_APPROVALS
        },
    }
    approvals["approvals"]["embedding_provenance"]["manifest_id"] = manifest_id

    policy = {
        "approved": True,
        "timezone": "UTC",
        "timestamp_mode": "offset_aware",
        "cutoff_inclusive": True,
        "late_arrival_policy": "exclude",
    }

    return [sensor], logs, approvals, policy


def _cutoff_fuse(sensor, logs, approvals, policy):
    return build_fusion_samples(
        sensor, logs,
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
        genuine_approval=approvals,
        point_in_time_policy=policy,
        has_log_feature="include",
        prediction_cutoff={(21, 6): "2025-01-01T11:00:00Z"},
    )


def test_timestamp_cutoff_excludes_late_log_but_keeps_audit_and_eligible_embedding():
    sensor, logs, approvals, policy = _synthetic_timestamp_records([
        ("synthetic-eligible", "synthetic-source-eligible", "2025-01-01T10:30:00Z", [2.0] * 768),
        ("synthetic-late", "synthetic-source-late", "2025-01-01T11:30:00Z", [100.0] * 768),
    ])

    result = _cutoff_fuse(sensor, logs, approvals, policy)

    assert result["text_embeddings"].shape == (1, 768)
    assert result["text_embeddings"][0, 0].item() == 2.0
    assert result["has_log"].tolist() == [True]
    assert result["log_ids_by_window"] == [["synthetic-eligible"]]
    assert [record["log_id"] for record in result["excluded_logs"]] == ["synthetic-late"]
    excluded = result["excluded_logs"][0]
    assert excluded["source_record_id"] == "synthetic-source-late"
    assert excluded["raw_timestamps"]["availability_time"] == "2025-01-01T11:30:00Z"


def test_malformed_embedding_is_rejected_even_when_timestamp_excludes_record():
    sensor, logs, approvals, policy = _synthetic_timestamp_records([
        ("synthetic-late", "synthetic-source-late", "2025-01-01T11:30:00Z", [float("nan")] * 768),
    ])

    with pytest.raises(ValueError, match="Non-finite Text embedding"):
        _cutoff_fuse(sensor, logs, approvals, policy)


def test_adapter_and_loader_fail_closed_for_duplicate_ids_dimensions_nonfinite_and_splits():
    duplicate_rows = [_metadata(0, "same-id", 1, 1), _metadata(1, "same-id", 1, 2)]
    with pytest.raises(ValueError, match="duplicate log_id"):
        _adapt(duplicate_rows, torch.zeros((2, 768)))

    valid_rows = [_metadata(0, "synthetic-id", 1, 1)]
    with pytest.raises(ValueError, match="shape"):
        _adapt(valid_rows, torch.zeros((1, 767)))
    with pytest.raises(ValueError, match="non-finite"):
        _adapt(valid_rows, torch.full((1, 768), float("inf")))
    with pytest.raises(ValueError, match="must be 'fixture'"):
        _adapt([{**valid_rows[0], "source_split": "train"}], torch.zeros((1, 768)))

    adapted = _adapt(valid_rows, torch.zeros((1, 768)))
    with pytest.raises(ValueError, match="Split mismatch"):
        _fuse([_sensor(1, 1, split="validation")], adapted)


def test_unmatched_log_raises_auditable_error_instead_of_silent_drop():
    rows = [_metadata(0, "synthetic-unmatched", 99, 4, source_record_id="synthetic-source")]
    records = _adapt(rows, torch.full((1, 768), 1.0))

    with pytest.raises(UnmatchedTextRecordsError) as caught:
        _fuse([_sensor(1, 4)], records)

    unmatched, = caught.value.unmatched_records
    assert unmatched["log_id"] == "synthetic-unmatched"
    assert unmatched["unit_id"] == 99
    assert unmatched["record_metadata"]["source_record_id"] == "synthetic-source"
    assert "text_embedding" not in unmatched["record_metadata"]
