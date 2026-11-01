"""Fail-closed multimodal data-contract compatibility layer.

This module restores the existing loader/dataset interface used by the fusion
contract tests. It intentionally does not train a model or create artifacts;
genuine multimodal training remains blocked until the approved data, lineage,
availability, and training authorizations exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from training.fusion_provenance import (
    TRAINING_AUTHORIZATION,
    validate_genuine_approval,
    validate_named_approval,
    validate_operational_horizon,
    validate_reproducibility_contract,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_integer(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid integer {field}: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid integer {field}: {value!r}") from exc
    if not isinstance(value, str) and value != parsed:
        raise ValueError(f"Non-integral {field}: {value!r}")
    return parsed


def validate_embedding_index_mapping(indices, embedding_count: int) -> list[int]:
    if isinstance(embedding_count, bool) or not isinstance(embedding_count, int) or embedding_count < 0:
        raise ValueError("embedding_count must be a non-negative integer.")
    parsed = [_metadata_integer(value, "embedding_index") for value in indices]
    if len(parsed) < embedding_count:
        missing = sorted(set(range(embedding_count)) - set(parsed))
        raise ValueError(f"Missing embedding_index values (metadata/embedding count mismatch): {missing[:5]}.")
    if len(parsed) > embedding_count:
        raise ValueError("Sensor embedding and metadata row counts differ.")
    if len(parsed) != len(set(parsed)):
        raise ValueError("Duplicate embedding_index values in sensor metadata.")
    invalid = [index for index in parsed if index < 0 or index >= embedding_count]
    if invalid:
        raise ValueError(f"Out-of-range embedding_index values: {invalid[:5]}.")
    missing = sorted(set(range(embedding_count)) - set(parsed))
    if missing:
        raise ValueError(f"Missing embedding_index values: {missing[:5]}.")
    return parsed


def load_records(sensor_dir: Path, text_file: Path, *, approval_config: dict, split_config: dict):
    """Load and validate genuine sensor/text records without writing artifacts."""
    if not isinstance(approval_config.get("approval_register_id"), str) or not approval_config["approval_register_id"].strip():
        raise ValueError("Genuine-data use is blocked without an approval_register_id.")
    validate_genuine_approval(approval_config)
    validate_named_approval(approval_config, TRAINING_AUTHORIZATION)
    for field in ("data_schema", "embedding_model", "embedding_revision", "prediction_cutoff", "point_in_time_policy"):
        if not approval_config.get(field):
            raise ValueError(f"Approved run metadata is missing {field}.")
    validate_operational_horizon(approval_config.get("operational_horizon"))
    validate_reproducibility_contract(approval_config)
    policies = approval_config.get("fusion_policies", {})
    if policies.get("missing_log") not in {"zero_vector", "error"}:
        raise ValueError("Approved fusion_policies.missing_log is required.")
    if policies.get("multiple_logs") not in {"mean_sorted_log_id", "error"}:
        raise ValueError("Approved fusion_policies.multiple_logs is required.")
    if approval_config.get("has_log_feature") not in {"include", "exclude"}:
        raise ValueError("Approved has_log_feature selection is required.")

    import pandas as pd

    train_ids = set(map(int, split_config.get("train_engine_ids", [])))
    val_ids = set(map(int, split_config.get("validation_engine_ids", [])))
    if not train_ids or not val_ids or train_ids & val_ids or len(train_ids) != split_config.get("train_engine_count") or len(val_ids) != split_config.get("validation_engine_count"):
        raise ValueError("Frozen split manifest is missing, inconsistent, or has overlapping engine IDs.")
    metadata_path = sensor_dir / "sensor_embedding_metadata.csv"
    metadata_sha_before = sha256_file(metadata_path)
    split_metadata = pd.read_csv(metadata_path, usecols=["source_split", "unit_id"])
    if not split_metadata["source_split"].isin(["train", "validation"]).all():
        raise ValueError("Unexpected split found in sensor metadata. Only train and validation embeddings are permitted.")
    guarded_ids = []
    for raw_id, source_split in zip(split_metadata["unit_id"], split_metadata["source_split"]):
        unit_id = _metadata_integer(raw_id, "unit_id")
        if unit_id <= 0:
            raise ValueError(f"Invalid unit_id in sensor metadata: {raw_id!r}")
        guarded_ids.append(unit_id)
        expected = "train" if unit_id in train_ids else "validation" if unit_id in val_ids else None
        if expected is None or source_split != expected:
            raise ValueError(f"Sensor metadata split for unit {unit_id} is not authorized by the frozen train/validation equipment manifest.")
    if not isinstance(approval_config["prediction_cutoff"], dict):
        raise ValueError("Approved prediction_cutoff must map every window key to its cutoff.")
    metadata = pd.read_csv(metadata_path)
    if sha256_file(metadata_path) != metadata_sha_before:
        raise ValueError("Sensor metadata changed while split and target data were read.")
    required = {"unit_id", "window_start_cycle", "window_end_cycle", "label_cycle", "label_rul", "source_split", "namespace", "embedding_index", "source_dataset", "source_dataset_version", "source_record_id", "equipment_mapping_id", "preprocessing_version", "artifact_id", "embedding_model", "embedding_revision"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Sensor artifact lacks authoritative provenance columns: {sorted(missing)}. Genuine loading is blocked.")
    metadata_ids = [_metadata_integer(value, "unit_id") for value in metadata["unit_id"]]
    if metadata["source_split"].tolist() != split_metadata["source_split"].tolist() or metadata_ids != guarded_ids:
        raise ValueError("Sensor metadata changed between split and target reads.")
    if not metadata["namespace"].eq("genuine").all():
        raise ValueError("Genuine training requires explicit genuine sensor namespaces.")
    metadata["unit_id"] = metadata_ids
    for field in ("label_cycle", "window_start_cycle", "window_end_cycle", "embedding_index"):
        metadata[field] = [_metadata_integer(value, field) for value in metadata[field]]
    if ((metadata["window_start_cycle"] <= 0).any() or (metadata["window_end_cycle"] < metadata["window_start_cycle"]).any() or (metadata["label_cycle"] != metadata["window_end_cycle"]).any()):
        raise ValueError("Sensor window bounds must be valid and label_cycle must equal window_end_cycle.")
    metadata["label_rul"] = pd.to_numeric(metadata["label_rul"], errors="raise")
    if metadata["label_rul"].isna().any() or not np.isfinite(metadata["label_rul"].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite RUL labels in sensor metadata.")
    expected_rul = metadata.groupby("unit_id")["label_cycle"].transform("max") - metadata["label_cycle"]
    if not np.array_equal(metadata["label_rul"].to_numpy(dtype=float), expected_rul.to_numpy(dtype=float)):
        raise ValueError("Sensor artifact RUL labels do not match the frozen max_cycle(unit) - cycle target.")
    embeddings = np.load(sensor_dir / "sensor_embeddings.npy")
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(f"Unexpected sensor embedding shape: {embeddings.shape}")
    metadata["embedding_index"] = validate_embedding_index_mapping(metadata["embedding_index"].tolist(), int(embeddings.shape[0]))
    if metadata.duplicated(["unit_id", "label_cycle"]).any() or not np.isfinite(embeddings).all():
        raise ValueError("Duplicate sensor windows or non-finite sensor embeddings detected.")
    embedding_hash = sha256_file(sensor_dir / "sensor_embeddings.npy")
    sensor_records = []
    for row in metadata.to_dict("records"):
        sensor_records.append({"unit_id": int(row["unit_id"]), "cycle": int(row["label_cycle"]), "window_start_cycle": int(row["window_start_cycle"]), "window_end_cycle": int(row["window_end_cycle"]), "split": str(row["source_split"]), "sensor_embedding": embeddings[int(row["embedding_index"])], "rul": float(row["label_rul"]), "namespace": row["namespace"], **{field: row[field] for field in ("source_dataset", "source_dataset_version", "source_record_id", "equipment_mapping_id", "preprocessing_version", "artifact_id", "embedding_model", "embedding_revision")}, "artifact_sha256": embedding_hash})
    with text_file.open("r", encoding="utf-8") as handle:
        text_records = json.load(handle)
    if not isinstance(text_records, list):
        raise ValueError("Text JSON must contain a list of text records.")
    for record in text_records:
        if record.get("namespace") != "genuine":
            raise ValueError("Training accepts only explicitly namespaced genuine text records.")
        for field in ("data_schema", "embedding_model", "embedding_revision"):
            if record.get(field) != approval_config[field]:
                raise ValueError(f"Text record {record.get('log_id')!r} has missing or mismatched {field}.")
    return sensor_records, text_records


def make_dataset(samples, split_name: str, *, use_has_log: bool) -> TensorDataset:
    indices = [index for index, split in enumerate(samples["split"]) if split == split_name]
    if not indices:
        raise ValueError(f"No samples found for split {split_name!r}.")
    index = torch.tensor(indices, dtype=torch.long)
    tensors = [samples["sensor_embeddings"][index], samples["text_embeddings"][index], samples["rul"][index]]
    if use_has_log:
        tensors.append(samples["has_log"][index])
    return TensorDataset(*tensors)


def main() -> None:
    raise RuntimeError("Multimodal training is intentionally blocked; use the approved contract before adding a training entry point.")


if __name__ == "__main__":
    main()
