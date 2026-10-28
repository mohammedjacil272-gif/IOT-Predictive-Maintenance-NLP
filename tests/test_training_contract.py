from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import torch

from training.train_multimodal import (
    load_records,
    make_dataset,
    validate_embedding_index_mapping,
)
from training.fusion_provenance import GENUINE_APPROVALS, TRAINING_AUTHORIZATION


def approval_config():
    approved_record = {
        "status": "APPROVED",
        "approved_by": "fixture-test-only",
        "evidence_refs": ["synthetic-unit-test"],
    }
    return {
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
        "prediction_cutoff": {"1:1": "2025-01-01T00:00:00Z"},
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
    }


def test_official_test_split_is_rejected_before_rul_column_is_read(
    monkeypatch,
):
    monkeypatch.setattr(
        np,
        "load",
        lambda _path: np.zeros((1, 64), dtype=np.float32),
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
        lambda _path: np.zeros((1, 64), dtype=np.float32),
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


def test_sensor_target_must_match_frozen_uncapped_rul(monkeypatch):
    sensor_dir = Path("unused-sensor-dir")
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
        }
    )
    embeddings = np.zeros((2, 64), dtype=np.float32)
    embeddings[0, 0], embeddings[1, 0] = 20.0, 10.0
    original_open = Path.open

    monkeypatch.setattr(np, "load", lambda _path: embeddings)
    monkeypatch.setattr("training.train_multimodal.sha256_file", lambda _path: "a" * 64)
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
        approval_config=approval_config(),
        split_config=split_config,
    )
    assert [record["rul"] for record in records] == [1.0, 0.0]
    # embedding_index addresses vectors inside the sensor artifact only.
    assert records[0]["sensor_embedding"][0] == 10.0
    assert logs == []

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
            approval_config=approval_config(),
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
            approval_config=approval_config(),
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


def test_training_dataset_includes_has_log_only_when_selected():
    samples = {
        "split": ("train",),
        "sensor_embeddings": torch.zeros((1, 64)),
        "text_embeddings": torch.zeros((1, 768)),
        "rul": torch.tensor([1.0]),
        "has_log": torch.tensor([False]),
    }
    excluded = make_dataset(samples, "train", use_has_log=False)
    included = make_dataset(samples, "train", use_has_log=True)
    assert len(excluded.tensors) == 3
    assert len(included.tensors) == 4
    assert included.tensors[3].tolist() == [False]
