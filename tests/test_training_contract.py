
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import torch

from training.train_multimodal import (
    main as training_main,
    load_records,
    make_dataset,
    _load_sensor_embeddings,
    _load_text_embedding_artifact,
    _select_verified_text_embedding,
    validate_embedding_index_mapping,
)
from training.fusion_dataset import build_fusion_samples, embedding_digest
from training.fusion_provenance import GENUINE_APPROVALS, TRAINING_AUTHORIZATION


def approval_config(sensor_artifact_path=None):
    approved_record = {
        "status": "APPROVED",
        "approved_by": "fixture-test-only",
        "evidence_refs": ["synthetic-unit-test"],
    }
    manifest_id = "training-contract-manifest-v1"
    artifact_path = str(Path(__file__).resolve())
    sensor_artifact_path = str(sensor_artifact_path or artifact_path)
    artifact_hash = __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()
    config = {
        "approval_register_id": "synthetic-test-register",
        "approvals": {
            key: approved_record.copy()
            for key in (*GENUINE_APPROVALS, TRAINING_AUTHORIZATION)
        },
        "data_schema": "test-schema-v1",
        "embedding_model": "test-model",
        "embedding_revision": "test-revision",
        "preprocessing_version": "test-preprocessing-v1",
        "nlp_model_name": "test-nlp-model",
        "nlp_model_revision": "test-nlp-revision",
        "tokenizer_name": "test-tokenizer",
        "tokenizer_revision": "test-tokenizer-revision",
        "checkpoint_lineage": {
            "source_checkpoint_id": "synthetic-test-checkpoint",
            "source_checkpoint_sha256": "c" * 64,
        },
        "operational_horizon": {"value": 25, "unit": "cycles"},
        "prediction_cutoff": {
            "1:1": "2025-01-01T00:00:00Z",
            "1:2": "2025-01-01T00:00:00Z",
        },
        "point_in_time_policy": {
            "approved": True,
            "timezone": "UTC",
            "timestamp_mode": "offset_aware",
            "cutoff_inclusive": True,
            "late_arrival_policy": "reject",
        },
        "fusion_policies": {
            "missing_log": "zero_vector",
            "multiple_logs": "mean_sorted_log_id",
        },
        "has_log_feature": "exclude",
        "approved_manifest": {
            "manifest_id": manifest_id,
            "equipment_mapping_id": "test-map",
            "datasets": {
                "sensor": {
                    "source_dataset": "synthetic-test",
                    "source_dataset_version": "v1",
                    "preprocessing_version": "test-preprocess-v1",
                },
                "text": {
                    "source_dataset": "synthetic-test-log",
                    "source_dataset_version": "v1",
                    "preprocessing_version": "test-preprocessing-v1",
                },
            },
            "record_scope": {
                "sensor": {"scope_type": "dataset"},
                "text": {"scope_type": "dataset"},
            },
            "artifacts": {
                "test-embedding-artifact": {
                    "artifact_id": "test-embedding-artifact",
                    "artifact_sha256": __import__("hashlib").sha256(
                        Path(sensor_artifact_path).read_bytes()
                    ).hexdigest(),
                    "artifact_path": sensor_artifact_path,
                    "modality": "sensor",
                    "embedding_model": "test-model",
                    "embedding_revision": "test-revision",
                    "embedding_sha256_by_source_record_id": {
                        "row-1": embedding_digest([10.0] + [0.0] * 63),
                        "row-2": embedding_digest([20.0] + [0.0] * 63),
                    },
                },
                "test-text-artifact": {
                    "artifact_id": "test-text-artifact",
                    "artifact_sha256": artifact_hash,
                    "artifact_path": artifact_path,
                    "modality": "text",
                    "embedding_model": "test-model",
                    "embedding_revision": "test-revision",
                    "embedding_sha256": embedding_digest([0.0] * 768),
                },
            },
        },
    }
    config["approvals"]["embedding_provenance"]["manifest_id"] = manifest_id
    return config


def test_official_test_split_is_rejected_before_rul_column_is_read(
    monkeypatch,
):
    monkeypatch.setattr(
        np,
        "load",
        lambda _path, **kwargs: np.zeros((1, 64), dtype=np.float32),
    )
    reads = []

    def track_csv_columns(_path, *args, **kwargs):
        reads.append(kwargs.get("usecols"))
        return pd.DataFrame({"source_split": ["official_test"], "unit_id": [1]})

    monkeypatch.setattr(pd, "read_csv", track_csv_columns)
    monkeypatch.setattr("training.train_multimodal.sha256_file", lambda _path: "b" * 64)

    with pytest.raises(ValueError, match="Only train and validation"):
        load_records(
            Path("unused-sensor-dir"),
            Path("unused-text-records.json"),
            approval_config=approval_config(),
            split_config={
                "train_engine_ids": [1],
                "validation_engine_ids": [2],
                "train_engine_count": 1,
                "validation_engine_count": 1,
            },
        )

    assert reads == [["source_split", "unit_id"]]


def test_unlisted_equipment_is_rejected_before_rul_column_is_read(monkeypatch):
    monkeypatch.setattr(
        np,
        "load",
        lambda _path, **kwargs: np.zeros((1, 64), dtype=np.float32),
    )
    reads = []

    def track_csv_columns(_path, *args, **kwargs):
        reads.append(kwargs.get("usecols"))
        return pd.DataFrame({"source_split": ["validation"], "unit_id": [999]})

    monkeypatch.setattr(pd, "read_csv", track_csv_columns)
    monkeypatch.setattr("training.train_multimodal.sha256_file", lambda _path: "b" * 64)

    with pytest.raises(ValueError, match="not authorized by the frozen"):
        load_records(
            Path("unused-sensor-dir"),
            Path("unused-text-records.json"),
            approval_config=approval_config(),
            split_config={
                "train_engine_ids": [1],
                "validation_engine_ids": [2],
                "train_engine_count": 1,
                "validation_engine_count": 1,
            },
        )

    assert reads == [["source_split", "unit_id"]]


def test_existing_sensor_artifact_without_lineage_fails_closed(monkeypatch):
    metadata = pd.DataFrame(
        {
            "source_split": ["train"],
            "unit_id": [1],
            "label_cycle": [2],
            "label_rul": [0],
            "embedding_index": [0],
        }
    )
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda _path, *args, **kwargs: metadata[
            kwargs["usecols"]
        ].copy() if "usecols" in kwargs else metadata.copy(),
    )
    monkeypatch.setattr("training.train_multimodal.sha256_file", lambda _path: "b" * 64)
    monkeypatch.setattr(
        np, "load", lambda _path: pytest.fail("must reject before loading embeddings")
    )
    with pytest.raises(ValueError, match="lacks authoritative provenance columns"):
        load_records(
            Path("unused-sensor-dir"),
            Path("unused-text-records.json"),
            approval_config=approval_config(),
            split_config={
                "train_engine_ids": [1],
                "validation_engine_ids": [2],
                "train_engine_count": 1,
                "validation_engine_count": 1,
            },
        )


def test_sensor_target_must_match_frozen_uncapped_rul(monkeypatch, tmp_path):
    sensor_dir = tmp_path / "sensor"
    sensor_dir.mkdir()
    (sensor_dir / "sensor_embeddings.npy").write_bytes(b"approved sensor bytes")
    text_path = Path("logs.json")
    metadata = pd.DataFrame(
        {
            "source_split": ["train", "train"],
            "namespace": ["genuine", "genuine"],
            "unit_id": [1, 1],
            "window_start_cycle": [1, 2],
            "window_end_cycle": [1, 2],
            "label_cycle": [1, 2],
            "label_rul": [1.0, 0.0],
            "embedding_index": [1, 0],
            "source_dataset": ["synthetic-test", "synthetic-test"],
            "source_dataset_version": ["v1", "v1"],
            "source_record_id": ["row-1", "row-2"],
            "equipment_mapping_id": ["test-map", "test-map"],
            "preprocessing_version": ["test-preprocess-v1", "test-preprocess-v1"],
            "artifact_id": ["test-embedding-artifact", "test-embedding-artifact"],
            "embedding_model": ["test-model", "test-model"],
            "embedding_revision": ["test-revision", "test-revision"],
            "event_time": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "record_creation_time": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "ingestion_time": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "availability_time": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        }
    )
    embeddings = np.zeros((2, 64), dtype=np.float32)
    embeddings[0, 0], embeddings[1, 0] = 20.0, 10.0
    original_open = Path.open

    monkeypatch.setattr(np, "load", lambda _path, **kwargs: embeddings)
    manifest_hash = __import__("hashlib").sha256(
        (sensor_dir / "sensor_embeddings.npy").read_bytes()
    ).hexdigest()
    monkeypatch.setattr("training.train_multimodal.sha256_file", lambda _path: manifest_hash)
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda _path, *args, **kwargs: metadata[
            kwargs["usecols"]
        ].copy() if "usecols" in kwargs else metadata.copy(),
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda path, *args, **kwargs: StringIO("[]")
        if path.name == "logs.json"
        else original_open(path, *args, **kwargs),
    )

    split_config = {
        "train_engine_ids": [1],
        "validation_engine_ids": [2],
        "train_engine_count": 1,
        "validation_engine_count": 1,
    }
    records, logs = load_records(
        sensor_dir,
        text_path,
        approval_config=approval_config(sensor_dir / "sensor_embeddings.npy"),
        split_config=split_config,
    )
    assert [record["rul"] for record in records] == [1.0, 0.0]
    # embedding_index addresses vectors inside the sensor artifact only.
    assert records[0]["sensor_embedding"][0] == 10.0
    assert list(logs) == []

    fixture_metadata = metadata.copy()
    fixture_metadata["namespace"] = "synthetic_fixture"
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda _path, *args, **kwargs: fixture_metadata[
            kwargs["usecols"]
        ].copy() if "usecols" in kwargs else fixture_metadata.copy(),
    )
    with pytest.raises(ValueError, match="explicit genuine sensor namespaces"):
        load_records(
            sensor_dir,
            text_path,
            approval_config=approval_config(sensor_dir / "sensor_embeddings.npy"),
            split_config=split_config,
        )

    bad_metadata = metadata.copy()
    bad_metadata.loc[0, "label_rul"] = 99
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda _path, *args, **kwargs: bad_metadata[
            kwargs["usecols"]
        ].copy() if "usecols" in kwargs else bad_metadata.copy(),
    )
    with pytest.raises(ValueError, match=r"frozen max_cycle\(unit\) - cycle"):
        load_records(
            sensor_dir,
            text_path,
            approval_config=approval_config(sensor_dir / "sensor_embeddings.npy"),
            split_config=split_config,
        )


def test_embedding_indices_are_an_explicit_artifact_local_mapping():
    assert validate_embedding_index_mapping([1, 0], 2) == [1, 0]
    with pytest.raises(ValueError, match="Duplicate embedding_index"):
        validate_embedding_index_mapping([0, 0], 2)
    with pytest.raises(ValueError, match="Missing embedding_index"):
        validate_embedding_index_mapping([0], 2)
    with pytest.raises(ValueError, match="Out-of-range embedding_index"):
        validate_embedding_index_mapping([0, 2], 2)
    with pytest.raises(ValueError, match="row counts differ"):
        validate_embedding_index_mapping([0, 1, 2], 2)


def test_object_sensor_arrays_are_rejected_before_use(monkeypatch):
    monkeypatch.setattr(
        np,
        "load",
        lambda _path, **kwargs: np.array([[{"x": 1}]], dtype=object),
    )
    with pytest.raises(ValueError, match="Object arrays"):
        _load_sensor_embeddings(Path("object.npy"))


def _write_text_embedding_artifact(tmp_path, metadata_rows=None):
    artifact_path = tmp_path / "embeddings.pt"
    metadata_path = tmp_path / "embedding_metadata.csv"
    embeddings = torch.zeros((2, 768), dtype=torch.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 2.0
    torch.save(embeddings, artifact_path)
    pd.DataFrame(metadata_rows or [
        {"row_index": 0, "source_record_id": "source-a"},
        {"row_index": 1, "source_record_id": "source-b"},
    ]).to_csv(metadata_path, index=False)
    return artifact_path, metadata_path, embeddings


def test_genuine_text_vectors_are_loaded_from_verified_artifact_rows(tmp_path):
    artifact_path, metadata_path, embeddings = _write_text_embedding_artifact(tmp_path)
    records = _load_text_embedding_artifact(
        artifact_path,
        metadata_path,
        expected_sha256=__import__("hashlib").sha256(artifact_path.read_bytes()).hexdigest(),
    )
    assert torch.equal(records["source-a"], embeddings[0])
    assert torch.equal(records["source-b"], embeddings[1])


@pytest.mark.parametrize("source_id", [None, np.nan, "", "   ", "nan"])
def test_text_artifact_rejects_null_or_blank_source_record_ids(tmp_path, source_id):
    rows = [
        {"row_index": 0, "source_record_id": source_id},
        {"row_index": 1, "source_record_id": "source-b"},
    ]
    artifact_path, metadata_path, _ = _write_text_embedding_artifact(tmp_path, rows)
    with pytest.raises(ValueError, match="source_record_id"):
        _load_text_embedding_artifact(
            artifact_path,
            metadata_path,
            expected_sha256=__import__("hashlib").sha256(artifact_path.read_bytes()).hexdigest(),
        )


def test_text_artifact_rejects_complex_tensor_dtype(tmp_path):
    artifact_path, metadata_path, _ = _write_text_embedding_artifact(tmp_path)
    torch.save(torch.ones((2, 768), dtype=torch.complex64), artifact_path)
    with pytest.raises(ValueError, match="floating-point"):
        _load_text_embedding_artifact(
            artifact_path,
            metadata_path,
            expected_sha256=__import__("hashlib").sha256(artifact_path.read_bytes()).hexdigest(),
        )


def test_json_text_vector_substitution_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        _select_verified_text_embedding(
            {"text_embedding": torch.zeros(768)},
            torch.ones(768),
        )


@pytest.mark.parametrize("rows", [
    [
        {"row_index": 0, "source_record_id": "source-a"},
        {"row_index": 0, "source_record_id": "source-b"},
    ],
    [
        {"row_index": 0, "source_record_id": "source-a"},
    ],
    [
        {"row_index": 0, "source_record_id": "source-a"},
        {"row_index": 2, "source_record_id": "source-b"},
    ],
])
def test_text_artifact_row_mapping_must_be_unique_complete_and_in_bounds(tmp_path, rows):
    artifact_path, metadata_path, _ = _write_text_embedding_artifact(tmp_path, rows)
    with pytest.raises(ValueError, match="row|mapping"):
        _load_text_embedding_artifact(
            artifact_path,
            metadata_path,
            expected_sha256=__import__("hashlib").sha256(artifact_path.read_bytes()).hexdigest(),
        )


def test_text_artifact_bytes_must_match_approved_hash(tmp_path):
    artifact_path, metadata_path, _ = _write_text_embedding_artifact(tmp_path)
    approved_hash = __import__("hashlib").sha256(artifact_path.read_bytes()).hexdigest()
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="approved SHA-256"):
        _load_text_embedding_artifact(
            artifact_path, metadata_path, expected_sha256=approved_hash
        )


def test_training_dataset_includes_has_log_only_when_selected():
    with pytest.raises(ValueError, match="strict fusion validation"):
        make_dataset({"split": ("train",)}, "train", use_has_log=False)

    samples = build_fusion_samples(
        [{
            "unit_id": 1,
            "cycle": 1,
            "split": "train",
            "namespace": "synthetic_fixture",
            "sensor_embedding": [0.0] * 64,
            "rul": 1.0,
        }],
        [],
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
    )

    excluded = make_dataset(samples, "train", use_has_log=False)
    included = make_dataset(samples, "train", use_has_log=True)
    assert len(excluded.tensors) == 3
    assert len(included.tensors) == 4
    assert included.tensors[3].tolist() == [False]


def test_forged_validated_sample_marker_is_rejected():
    from training.fusion_dataset import ValidatedFusionSamples

    assert not hasattr(ValidatedFusionSamples, "from_validated")
    with pytest.raises(TypeError, match="must come from strict fusion validation"):
        ValidatedFusionSamples({"split": ("train",)})


def test_validated_samples_are_immutable_and_defensive():
    samples = build_fusion_samples(
        [{
            "unit_id": 1,
            "cycle": 1,
            "split": "train",
            "namespace": "synthetic_fixture",
            "sensor_embedding": [0.0] * 64,
            "rul": 1.0,
        }],
        [],
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
    )
    with pytest.raises(TypeError):
        samples["split"] = ("validation",)

    detached = samples["sensor_embeddings"]
    detached[0, 0] = 99.0
    assert float(samples["sensor_embeddings"][0, 0]) == 0.0


@pytest.mark.parametrize(
    "field",
    ["sensor_embeddings", "split", "sensor_provenance_by_window"],
)
def test_validation_receipt_rejects_post_validation_content_changes(field):
    from types import MappingProxyType

    samples = build_fusion_samples(
        [{
            "unit_id": 1,
            "cycle": 1,
            "split": "train",
            "namespace": "synthetic_fixture",
            "sensor_embedding": [0.0] * 64,
            "rul": 1.0,
        }],
        [],
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
    )
    changed = {key: samples[key] for key in samples}

    if field == "sensor_embeddings":
        changed[field] = changed[field].clone()
        changed[field][0, 0] = 1.0
    elif field == "split":
        changed[field] = ("validation",)
    else:
        changed[field] = ({"source_record_id": "substituted"},)

    object.__setattr__(samples, "_data", MappingProxyType(changed))

    with pytest.raises(ValueError, match="content digest mismatch"):
        make_dataset(samples, "train", use_has_log=False)


def test_training_entrypoint_remains_fail_closed():
    with pytest.raises(RuntimeError, match="intentionally blocked"):
        training_main()
