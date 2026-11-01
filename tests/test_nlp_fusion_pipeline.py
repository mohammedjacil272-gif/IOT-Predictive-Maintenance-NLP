"""End-to-end tests for the enforced NLP-to-fusion boundary."""

import csv
import json
from pathlib import Path

import pytest
import torch

import training.nlp_fusion_pipeline as pipeline
from training.nlp_artifact_validation import validate_nlp_artifact_package


def _write_package(root: Path, *, shape=(1, 768), entity_log_id="log-0") -> None:
    metadata = {
        "row_index": "0",
        "log_id": "log-0",
        "unit_id": "1",
        "cycle": "1",
        "source_split": "fixture",
        "model_name": "fixture-model",
        "tokenizer_name": "fixture-tokenizer",
        "model_revision": "fixture-model-revision",
        "tokenizer_revision": "fixture-tokenizer-revision",
        "max_length": "128",
        "pooling": "mean",
        "embedding_dim": "768",
        "source_system": "synthetic-test",
        "source_record_id": "record-0",
        "source_dataset": "synthetic-fixture",
        "source_dataset_version": "fixture-v1",
        "event_timestamp": "2025-01-01T10:00:00Z",
        "created_at": "2025-01-01T10:00:00Z",
        "ingested_at": "2025-01-01T10:00:00Z",
        "available_at": "2025-01-01T10:00:00Z",
    }
    torch.save(torch.zeros(shape), root / "embeddings.pt")
    with (root / "embedding_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata))
        writer.writeheader()
        writer.writerow(metadata)
    with (root / "processed_logs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["log_id", "unit_id", "cycle", "source_split"])
        writer.writeheader()
        writer.writerow({"log_id": "log-0", "unit_id": "1", "cycle": "1", "source_split": "fixture"})
    (root / "entities.jsonl").write_text(
        json.dumps({"log_id": entity_log_id, "label": "bearing"}) + "\n",
        encoding="utf-8",
    )
    with (root / "severity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["log_id", "severity_score"])
        writer.writeheader()
        writer.writerow({"log_id": "log-0", "severity_score": "0.5"})
    (root / "verification_report.json").write_text(
        json.dumps({
            "dataset_type": "synthetic_fixture",
            "embedding_shape": list(shape),
            "number_of_logs": shape[0],
            "embedding_finite": True,
            "official_test_evaluation_performed": False,
            "model_name": "fixture-model",
            "tokenizer_name": "fixture-tokenizer",
            "model_revision": "fixture-model-revision",
            "tokenizer_revision": "fixture-tokenizer-revision",
            "max_length": 128,
            "pooling": "mean",
        }),
        encoding="utf-8",
    )
    (root / "artifact_manifest.json").write_text(
        json.dumps({"dataset_type": "synthetic_fixture"}), encoding="utf-8"
    )


def _sensor_records(*, cycles=(1,)):
    return [
        {
            "unit_id": 1,
            "cycle": cycle,
            "split": "fixture",
            "sensor_embedding": [0.0] * 64,
            "rul": 10.0,
            "namespace": "synthetic_fixture",
        }
        for cycle in cycles
    ]


def test_valid_synthetic_package_reaches_adapter_and_preserves_provenance(tmp_path):
    _write_package(tmp_path)
    result = pipeline.build_validated_fusion_samples(
        tmp_path,
        _sensor_records(),
        missing_log_policy="error",
        multiple_log_policy="error",
    )
    assert result["validation"]["status"] == "PASS"
    assert result["validation"]["genuine_data_claim"] is False
    assert result["source_report"]["dataset_type"] == "synthetic_fixture"
    assert result["samples"]["log_ids_by_window"] == [["log-0"]]
    assert result["samples"]["entities_by_window"][0][0]["log_id"] == "log-0"
    assert result["samples"]["text_embeddings"].shape == (1, 768)


def test_invalid_package_never_reaches_adapter(tmp_path, monkeypatch):
    _write_package(tmp_path, shape=(1, 767))
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("adapter must not receive failed validation")

    monkeypatch.setattr(pipeline, "adapt_nlp_embedding_metadata", fail_if_called)
    with pytest.raises(ValueError, match="validation did not PASS"):
        pipeline.build_validated_fusion_samples(
            tmp_path,
            _sensor_records(),
            missing_log_policy="error",
            multiple_log_policy="error",
        )
    assert called is False


def test_invalid_entity_ownership_is_rejected_before_fusion(tmp_path):
    _write_package(tmp_path, entity_log_id="unknown-log")
    with pytest.raises(ValueError, match="validation did not PASS"):
        pipeline.build_validated_fusion_samples(
            tmp_path,
            _sensor_records(),
            missing_log_policy="error",
            multiple_log_policy="error",
        )


def test_failed_validation_cannot_produce_samples(tmp_path):
    _write_package(tmp_path)
    report = json.loads((tmp_path / "verification_report.json").read_text(encoding="utf-8"))
    report["embedding_shape"] = [99, 768]
    (tmp_path / "verification_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="validation did not PASS"):
        pipeline.build_validated_fusion_samples(
            tmp_path,
            _sensor_records(),
            missing_log_policy="error",
            multiple_log_policy="error",
        )


def test_existing_missing_log_policy_remains_explicit(tmp_path):
    _write_package(tmp_path)
    with pytest.raises(ValueError, match="Missing technician log"):
        pipeline.build_validated_fusion_samples(
            tmp_path,
            _sensor_records(cycles=(1, 2)),
            missing_log_policy="error",
            multiple_log_policy="error",
        )
    result = pipeline.build_validated_fusion_samples(
        tmp_path,
        _sensor_records(cycles=(1, 2)),
        missing_log_policy="zero_vector",
        multiple_log_policy="error",
    )
    assert result["samples"]["has_log"].tolist() == [True, False]
