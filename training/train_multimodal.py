
"""Fail-closed multimodal data-contract compatibility layer.

This module restores the existing loader/dataset interface used by the fusion
contract tests. It intentionally does not train a model or create artifacts;
genuine multimodal training remains blocked until the approved data, lineage,
availability, and training authorizations exist.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import TensorDataset

from training.fusion_provenance import (
    TRAINING_AUTHORIZATION,
    TIMESTAMP_FIELDS,
    validate_genuine_approval,
    validate_named_approval,
    validate_operational_horizon,
    validate_reproducibility_contract,
    validate_point_in_time,
    validate_record_against_manifest,
)
from training.fusion_dataset import (
    ValidatedFusionSamples,
    _VALIDATED_INSTANCES,
    canonical_content_digest,
    embedding_digest,
    validate_embedding_against_manifest,
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


def _freeze_record(value: object) -> object:
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous()
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_record(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_record(item) for item in value)
    return value


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


def _load_sensor_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(path, allow_pickle=False)
    if embeddings.dtype.kind == "O":
        raise ValueError("Object arrays are not permitted for sensor embeddings.")
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(f"Unexpected sensor embedding shape: {embeddings.shape}")
    if not np.issubdtype(embeddings.dtype, np.floating) or not np.isfinite(embeddings).all():
        raise ValueError("Sensor embeddings must be finite floating-point values.")
    return embeddings


def _load_text_embedding_artifact(
    artifact_path: Path,
    metadata_path: Path,
    *,
    expected_sha256: str,
) -> dict[str, torch.Tensor]:
    """Load a verified text tensor and its artifact-local source-record mapping."""
    if not artifact_path.is_file() or sha256_file(artifact_path).lower() != expected_sha256.lower():
        raise ValueError("Text embedding artifact bytes do not match the approved SHA-256.")
    try:
        embeddings = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Unable to safely load text embedding artifact: {artifact_path}") from exc
    if isinstance(embeddings, dict) and "embeddings" in embeddings:
        embeddings = embeddings["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("Text embedding artifact must contain a tensor.")
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError(f"Unexpected text embedding artifact shape: {tuple(embeddings.shape)}")
    if not embeddings.dtype.is_floating_point:
        raise ValueError("Text embedding artifact must contain floating-point values.")
    embeddings = embeddings.to(torch.float32).contiguous()
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("Text embedding artifact contains non-finite values.")

    import pandas as pd

    try:
        metadata = pd.read_csv(metadata_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read text embedding metadata: {metadata_path}") from exc
    required = {"row_index", "source_record_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Text embedding metadata is missing columns: {sorted(missing)}")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError("Text embedding metadata row count does not match the artifact tensor.")
    try:
        row_indices = [_metadata_integer(value, "row_index") for value in metadata["row_index"]]
    except ValueError as exc:
        raise ValueError("Text embedding metadata row_index values are invalid.") from exc
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("Duplicate text embedding row mappings are not permitted.")
    if set(row_indices) != set(range(embeddings.shape[0])):
        raise ValueError("Text embedding row mappings must be complete and in bounds.")
    source_ids = []
    for value in metadata["source_record_id"]:
        if pd.isna(value):
            raise ValueError("Text embedding source_record_id values must be non-null.")
        source_id = str(value).strip()
        if not source_id or source_id.lower() == "nan":
            raise ValueError("Text embedding source_record_id values must be non-empty.")
        source_ids.append(source_id)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Text embedding source_record_id mappings must be unique and non-empty.")
    return {source_id: embeddings[row_index].clone() for source_id, row_index in zip(source_ids, row_indices)}


def _select_verified_text_embedding(record: dict, loaded_embedding: torch.Tensor) -> torch.Tensor:
    """Use the artifact row as truth; compatibility JSON must match exactly."""
    supplied_embedding = record.get("text_embedding")
    if supplied_embedding is not None:
        try:
            supplied_tensor = torch.as_tensor(supplied_embedding, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Supplied text embedding is malformed.") from exc
        if supplied_tensor.shape != loaded_embedding.shape or not torch.equal(
            supplied_tensor, loaded_embedding
        ):
            raise ValueError("Supplied text embedding does not match the approved artifact row.")
    return loaded_embedding.clone()


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
    required = {"unit_id", "window_start_cycle", "window_end_cycle", "label_cycle", "label_rul", "source_split", "namespace", "embedding_index", "source_dataset", "source_dataset_version", "source_record_id", "equipment_mapping_id", "preprocessing_version", "artifact_id", "embedding_model", "embedding_revision", *TIMESTAMP_FIELDS}
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
    embeddings = _load_sensor_embeddings(sensor_dir / "sensor_embeddings.npy")
    metadata["embedding_index"] = validate_embedding_index_mapping(metadata["embedding_index"].tolist(), int(embeddings.shape[0]))
    if metadata.duplicated(["unit_id", "label_cycle"]).any() or not np.isfinite(embeddings).all():
        raise ValueError("Duplicate sensor windows or non-finite sensor embeddings detected.")
    embedding_hash = sha256_file(sensor_dir / "sensor_embeddings.npy")
    sensor_records = []
    for row in metadata.to_dict("records"):
        vector = embeddings[int(row["embedding_index"])]
        sensor_records.append({"unit_id": int(row["unit_id"]), "cycle": int(row["label_cycle"]), "window_start_cycle": int(row["window_start_cycle"]), "window_end_cycle": int(row["window_end_cycle"]), "split": str(row["source_split"]), "sensor_embedding": vector, "embedding_sha256": embedding_digest(vector), "rul": float(row["label_rul"]), "namespace": row["namespace"], **{field: row[field] for field in ("source_dataset", "source_dataset_version", "source_record_id", "equipment_mapping_id", "preprocessing_version", "artifact_id", "embedding_model", "embedding_revision", *TIMESTAMP_FIELDS)}, "artifact_sha256": embedding_hash, "artifact_path": str(sensor_dir / "sensor_embeddings.npy")})
    for record in sensor_records:
        validate_record_against_manifest(record, approval=approval_config, modality="sensor")
        validate_embedding_against_manifest(
            record,
            record["sensor_embedding"],
            approval_config["approved_manifest"],
            "sensor",
        )
        cutoff = approval_config["prediction_cutoff"].get(
            f"{record['unit_id']}:{record['cycle']}"
        ) or approval_config["prediction_cutoff"].get(
            (record["unit_id"], record["cycle"])
        )
        parsed = validate_point_in_time(
            record, cutoff=cutoff, policy=approval_config["point_in_time_policy"]
        )
        if parsed["eligible"] is False:
            raise ValueError("Sensor availability is after the approved prediction cutoff.")
    with text_file.open("r", encoding="utf-8") as handle:
        text_records = json.load(handle)
    if not isinstance(text_records, list):
        raise ValueError("Text JSON must contain a list of text records.")
    text_embedding_cache = {}
    for record in text_records:
        if record.get("namespace") != "genuine":
            raise ValueError("Training accepts only explicitly namespaced genuine text records.")
        for field in ("data_schema", "embedding_model", "embedding_revision"):
            if record.get(field) != approval_config[field]:
                raise ValueError(f"Text record {record.get('log_id')!r} has missing or mismatched {field}.")
        required_text_provenance = (
            "source_dataset", "source_dataset_version", "source_record_id",
            "equipment_mapping_id", "preprocessing_version", "artifact_id",
            "artifact_sha256", "artifact_path", "event_time",
            "record_creation_time", "ingestion_time", "availability_time",
        )
        missing = [field for field in required_text_provenance if not record.get(field)]
        if missing:
            raise ValueError(f"Text record {record.get('log_id')!r} has missing provenance: {missing}.")
        if sha256_file(Path(record["artifact_path"])) != record["artifact_sha256"].lower():
            raise ValueError(f"Text record {record.get('log_id')!r} artifact hash mismatch.")
        validate_point_in_time(
            record,
            cutoff=approval_config["prediction_cutoff"].get(
                f"{record.get('unit_id')}:{record.get('cycle')}"
            ) or approval_config["prediction_cutoff"].get(
                (record.get("unit_id"), record.get("cycle"))
            ),
            policy=approval_config["point_in_time_policy"],
        )
        validate_record_against_manifest(record, approval=approval_config, modality="text")
        artifact = approval_config["approved_manifest"]["artifacts"][record["artifact_id"]]
        artifact_path = Path(artifact["artifact_path"])
        metadata_path = Path(
            artifact.get("embedding_metadata_path")
            or artifact.get("metadata_path")
            or artifact_path.with_name("embedding_metadata.csv")
        )
        cache_key = (str(artifact_path), str(metadata_path))
        if cache_key not in text_embedding_cache:
            text_embedding_cache[cache_key] = _load_text_embedding_artifact(
                artifact_path,
                metadata_path,
                expected_sha256=artifact["artifact_sha256"],
            )
        loaded_embedding = text_embedding_cache[cache_key].get(record["source_record_id"])
        if loaded_embedding is None:
            raise ValueError(
                f"Text artifact has no embedding row for source_record_id {record['source_record_id']!r}."
            )
        record["text_embedding"] = _select_verified_text_embedding(record, loaded_embedding)
        if not isinstance(record.get("log_id"), str) or not record["log_id"].strip():
            raise ValueError("Genuine text record requires a non-empty log_id.")
        entities = record.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("Genuine text entities must be a list.")
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("Genuine text entity must be a mapping.")
            for field in ("log_id", "unit_id", "cycle", "split"):
                if entity.get(field) != record.get(field):
                    raise ValueError(f"Genuine text entity {field} does not match its record.")
        record["embedding_sha256"] = embedding_digest(record["text_embedding"])
        validate_embedding_against_manifest(
            record,
            record["text_embedding"],
            approval_config["approved_manifest"],
            "text",
        )
    return tuple(_freeze_record(record) for record in sensor_records), tuple(
        _freeze_record(record) for record in text_records
    )


def make_dataset(samples, split_name: str, *, use_has_log: bool) -> TensorDataset:
    if (
        not isinstance(samples, ValidatedFusionSamples)
        or _VALIDATED_INSTANCES.get(id(samples), lambda: None)() is not samples
    ):
        raise ValueError("Dataset construction requires samples returned by strict fusion validation.")
    expected_digest = canonical_content_digest(
        {
            "manifest_id": samples.manifest_id,
            "samples": {key: samples[key] for key in samples},
        }
    )
    if samples.content_digest != expected_digest:
        raise ValueError("Validated sample content digest mismatch.")
    if not isinstance(use_has_log, bool):
        raise ValueError("use_has_log must be an explicit boolean.")
    required = {
        "namespace", "sensor_embeddings", "text_embeddings", "rul", "has_log", "split",
        "unit_id", "cycle", "sensor_provenance_by_window", "sensor_timestamps_by_window",
    }
    missing = required - set(samples)
    if missing:
        raise ValueError(f"Validated samples are incomplete: {sorted(missing)}")
    namespace = samples["namespace"]
    if namespace == "genuine" or (isinstance(namespace, str) and namespace.startswith("genuine:")):
        if not samples.manifest_id:
            raise ValueError("Genuine samples are missing their approved manifest identity.")
    sensor = samples["sensor_embeddings"]
    text = samples["text_embeddings"]
    rul = samples["rul"]
    has_log = samples["has_log"]
    split_values = samples["split"]
    if sensor.ndim != 2 or sensor.shape[1] != 64 or text.ndim != 2 or text.shape[1] != 768:
        raise ValueError("Validated sample embedding dimensions are invalid.")
    if sensor.shape[0] != text.shape[0] or sensor.shape[0] != rul.shape[0] or sensor.shape[0] != len(split_values):
        raise ValueError("Validated sample lengths do not agree.")
    if not all(isinstance(item, str) and item.strip() for item in split_values):
        raise ValueError("Validated sample splits are malformed.")
    if not all(torch.isfinite(tensor).all().item() for tensor in (sensor, text, rul)):
        raise ValueError("Validated samples contain non-finite values.")
    approved_mode = samples.get("approved_has_log_feature")
    if approved_mode is not None and (approved_mode == "include") != use_has_log:
        raise ValueError("use_has_log does not match the approved fusion policy.")
    indices = [index for index, split in enumerate(samples["split"]) if split == split_name]
    if not indices:
        raise ValueError(f"No samples found for split {split_name!r}.")
    index = torch.tensor(indices, dtype=torch.long)
    tensors = [samples["sensor_embeddings"][index], samples["text_embeddings"][index], samples["rul"][index]]
    if use_has_log:
        tensors.append(samples["has_log"][index])
    return TensorDataset(*tensors)



def main() -> None:
    raise RuntimeError(
        "Multimodal training is intentionally blocked; "
        "use the approved contract before adding a training entry point."
    )


if __name__ == "__main__":
    main()
