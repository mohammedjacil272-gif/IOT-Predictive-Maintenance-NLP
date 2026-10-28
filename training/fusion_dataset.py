"""Metadata-based fusion alignment with explicit, provisional policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import re
from typing import Any

import torch

from training.fusion_provenance import (
    TIMESTAMP_FIELDS,
    validate_genuine_approval,
    validate_point_in_time,
)


def _validate_namespace(records: Sequence[Mapping], kind: str) -> str | None:
    if not records:
        return None
    namespaces = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{kind} record must be a metadata mapping.")
        namespace = record.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError(f"{kind} record is missing an explicit namespace.")
        namespaces.add(namespace.strip())
    if len(namespaces) != 1:
        raise ValueError(f"Ambiguous {kind} namespaces: {sorted(namespaces)}")
    namespace = next(iter(namespaces))
    if namespace not in {"synthetic_fixture", "genuine"} and not namespace.startswith(
        ("synthetic:", "genuine:")
    ):
        raise ValueError(f"Unsupported record namespace: {namespace!r}")
    return namespace


def _finite_vector(value: object, dimension: int, context: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Malformed {context} embedding.") from exc
    if tensor.shape != (dimension,):
        raise ValueError(f"{context} embedding must have {dimension} values.")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Non-finite {context} embedding.")
    return tensor


def _validate_genuine_artifact_provenance(record: Mapping, context: str) -> None:
    required = (
        "source_dataset",
        "source_dataset_version",
        "source_record_id",
        "equipment_mapping_id",
        "preprocessing_version",
        "artifact_id",
        "artifact_sha256",
        "embedding_model",
        "embedding_revision",
    )
    missing = [
        field for field in required
        if not isinstance(record.get(field), str) or not record[field].strip()
    ]
    if missing:
        raise ValueError(
            f"Genuine {context} record is missing provenance: {missing}"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", record["artifact_sha256"]):
        raise ValueError(f"Genuine {context} artifact_sha256 must be a SHA-256 hex digest.")


SENSOR_PROVENANCE_FIELDS = (
    "source_dataset", "source_dataset_version", "source_record_id",
    "equipment_mapping_id", "preprocessing_version", "artifact_id",
    "artifact_sha256", "embedding_model", "embedding_revision",
    "window_start_cycle", "window_end_cycle",
)
LOG_PROVENANCE_FIELDS = (
    "source_dataset", "source_dataset_version", "source_system",
    "source_record_id", "equipment_mapping_id", "preprocessing_version",
    "artifact_id", "artifact_sha256", "embedding_model", "embedding_revision",
)


def _window_cutoff(cutoff: object, key: tuple[int, int]) -> object:
    """Resolve a required cutoff for one prediction window."""
    if isinstance(cutoff, Mapping):
        for candidate in (key, f"{key[0]}:{key[1]}"):
            if candidate in cutoff:
                return cutoff[candidate]
        raise ValueError(f"Missing prediction cutoff for sensor window {key}.")
    return cutoff


def build_fusion_samples(
    sensor_records: Sequence[Mapping],
    text_records: Sequence[Mapping],
    *,
    missing_log_policy: str | None = None,
    multiple_log_policy: str | None = None,
    genuine_approval: Mapping | None = None,
    point_in_time_policy: Mapping | None = None,
    prediction_cutoff: object | None = None,
) -> dict[str, Any]:
    """Join by canonical `(unit_id, cycle)` metadata, never record position.

    Policy arguments are mandatory and intentionally have no production
    defaults. Supported provisional choices are `zero_vector`/`error` for
    missing logs and `mean_sorted_log_id`/`error` for multiple logs.
    """
    if missing_log_policy not in {"zero_vector", "error"}:
        raise ValueError("Select missing_log_policy explicitly: zero_vector or error.")
    if multiple_log_policy not in {"mean_sorted_log_id", "error"}:
        raise ValueError(
            "Select multiple_log_policy explicitly: mean_sorted_log_id or error."
        )

    sensor_namespace = _validate_namespace(sensor_records, "Sensor")
    text_namespace = _validate_namespace(text_records, "Text")
    if sensor_namespace is None:
        raise ValueError("No sensor records supplied.")
    namespace = sensor_namespace
    if text_namespace is not None and text_namespace != namespace:
        raise ValueError(
            f"Sensor/text namespace mismatch: {namespace!r} vs {text_namespace!r}."
        )
    is_genuine = namespace == "genuine" or namespace.startswith("genuine:")
    if is_genuine:
        validate_genuine_approval(genuine_approval)
        if point_in_time_policy is None or not isinstance(prediction_cutoff, Mapping):
            raise ValueError(
                "Genuine-data use is blocked without an approved point-in-time policy and per-window cutoffs."
            )

    sensor_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    split_by_unit: dict[int, str] = {}
    for record in sensor_records:
        required = {"unit_id", "cycle", "split", "sensor_embedding", "rul", "namespace"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Sensor record is missing fields: {sorted(missing)}")
        if is_genuine:
            _validate_genuine_artifact_provenance(record, "sensor")
            if (
                not isinstance(record.get("window_start_cycle"), int)
                or not isinstance(record.get("window_end_cycle"), int)
                or record["window_start_cycle"] <= 0
                or record["window_end_cycle"] < record["window_start_cycle"]
                or record["window_end_cycle"] != record.get("cycle")
            ):
                raise ValueError("Genuine sensor window bounds must match its canonical cycle.")
        unit_id, cycle = record["unit_id"], record["cycle"]
        if not isinstance(unit_id, int) or isinstance(unit_id, bool) or unit_id <= 0:
            raise ValueError(f"Sensor unit_id must be a positive integer: {unit_id!r}")
        if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle <= 0:
            raise ValueError(f"Sensor cycle must be a positive integer: {cycle!r}")
        split = record["split"]
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"Sensor split must be a non-empty string for {(unit_id, cycle)}.")
        if split != split.strip():
            raise ValueError(f"Sensor split contains surrounding whitespace for {(unit_id, cycle)}.")
        old_split = split_by_unit.setdefault(unit_id, split)
        if old_split != split:
            raise ValueError(f"Equipment split leakage for unit {unit_id}: {old_split!r} and {split!r}.")
        key = (unit_id, cycle)
        if key in sensor_by_key:
            raise ValueError(f"Duplicate sensor equipment/cycle pair: {key}")
        embedding = _finite_vector(record["sensor_embedding"], 64, "Sensor")
        try:
            if isinstance(record["rul"], bool):
                raise TypeError
            rul = float(record["rul"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed supplied RUL for {key}.") from exc
        if not torch.isfinite(torch.tensor(rul)).item():
            raise ValueError(f"Non-finite supplied RUL for {key}.")
        if rul < 0:
            raise ValueError(f"Supplied RUL must be non-negative for {key}.")
        sensor_by_key[key] = {
            "embedding": embedding,
            "rul": rul,
            "split": split,
            "provenance": {
                field: record.get(field) for field in SENSOR_PROVENANCE_FIELDS
            },
        }

    if not sensor_by_key:
        raise ValueError("No sensor records supplied.")

    logs_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen_log_ids: set[str] = set()
    seen_source_ids: set[tuple[str, str]] = set()
    source_system_by_id: dict[str, str] = {}
    excluded_logs: list[dict[str, Any]] = []
    for record in text_records:
        required = {"log_id", "unit_id", "cycle", "split", "text_embedding", "namespace"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Text record is missing fields: {sorted(missing)}")
        if is_genuine:
            _validate_genuine_artifact_provenance(record, "technician-log")
        unit_id, cycle = record["unit_id"], record["cycle"]
        if not isinstance(unit_id, int) or isinstance(unit_id, bool) or unit_id <= 0:
            raise ValueError(f"Text unit_id must be a positive integer: {unit_id!r}")
        if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle <= 0:
            raise ValueError(f"Text cycle must be a positive integer: {cycle!r}")
        split = record["split"]
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"Text split must be a non-empty string for {(unit_id, cycle)}.")
        if split != split.strip():
            raise ValueError(f"Text split contains surrounding whitespace for {(unit_id, cycle)}.")
        key = (unit_id, cycle)
        if key not in sensor_by_key:
            raise ValueError(f"Text record has no matching sensor window: {key}")
        if split != sensor_by_key[key]["split"]:
            raise ValueError(f"Split mismatch for {key}")
        log_id = record["log_id"]
        if not isinstance(log_id, str) or not log_id.strip():
            raise ValueError("Text record has a missing or empty log_id.")
        log_id = log_id.strip()
        if log_id in seen_log_ids:
            raise ValueError(f"Duplicate log_id: {log_id}")
        seen_log_ids.add(log_id)

        if is_genuine:
            for field in ("source_system", "source_record_id", *TIMESTAMP_FIELDS):
                if field not in record or record[field] in (None, ""):
                    raise ValueError(f"Genuine text record is missing {field}.")
            if not isinstance(record["source_system"], str) or not isinstance(
                record["source_record_id"], str
            ):
                raise ValueError("Genuine source-system provenance identifiers must be strings.")
            source_key = (
                record["source_system"].strip(),
                record["source_record_id"].strip(),
            )
            if not all(source_key):
                raise ValueError("Genuine source-system provenance must be non-empty.")
            if source_key in seen_source_ids:
                raise ValueError(f"Duplicate source-system record identifier: {source_key}")
            previous_system = source_system_by_id.setdefault(source_key[1], source_key[0])
            if previous_system != source_key[0]:
                raise ValueError(
                    f"Ambiguous source-system ID collision for {source_key[1]!r}: "
                    f"{previous_system!r} and {source_key[0]!r}."
                )
            seen_source_ids.add(source_key)
            parsed_times = validate_point_in_time(
                record,
                cutoff=_window_cutoff(prediction_cutoff, key),
                policy=point_in_time_policy,
            )
            if parsed_times["eligible"] is False:
                excluded_logs.append({
                    "log_id": log_id,
                    "source_system": source_key[0],
                    "source_record_id": source_key[1],
                    "unit_id": unit_id,
                    "cycle": cycle,
                    "split": split,
                    "timestamps": parsed_times,
                    "raw_timestamps": {
                        field: record[field] for field in TIMESTAMP_FIELDS
                    },
                    "source_provenance": {
                        field: record.get(field)
                        for field in LOG_PROVENANCE_FIELDS
                    },
                    "exclusion_reason": "availability after prediction cutoff",
                })
                continue
        else:
            parsed_times = {field: record.get(field) for field in TIMESTAMP_FIELDS}

        embedding = _finite_vector(record["text_embedding"], 768, "Text")
        entities = record.get("entities") or []
        if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
            raise ValueError(f"Entities for log {log_id} must be a sequence.")
        expected = {"log_id": log_id, "unit_id": unit_id, "cycle": cycle, "split": split}
        copied_entities = []
        for entity in entities:
            if not isinstance(entity, Mapping):
                raise ValueError(f"Each entity for log {log_id} must be a mapping.")
            item = dict(entity)
            if is_genuine:
                missing_entity_provenance = {
                    "log_id", "unit_id", "cycle", "split"
                } - item.keys()
                if missing_entity_provenance:
                    raise ValueError(
                        f"Genuine entity for log {log_id} is missing provenance: "
                        f"{sorted(missing_entity_provenance)}"
                    )
            for field, expected_value in expected.items():
                if field in item and item[field] != expected_value:
                    raise ValueError(f"Entity provenance mismatch for log {log_id}: {field}")
                item[field] = expected_value
            copied_entities.append(item)
        severity_score = record.get("severity_score")
        if severity_score is not None:
            try:
                severity_score = float(severity_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed severity score for log {log_id}.") from exc
            if not torch.isfinite(torch.tensor(severity_score)).item():
                raise ValueError(f"Non-finite severity score for log {log_id}.")
        logs_by_key[key].append({
            "log_id": log_id,
            "embedding": embedding,
            "entities": copied_entities,
            "severity": {
                **expected,
                "severity_score": severity_score,
                "severity_label": record.get("severity_label"),
                "matched_terms": record.get("matched_terms"),
                "explanation": record.get("explanation"),
            },
            "timestamps": parsed_times,
            "raw_timestamps": {
                field: record.get(field) for field in TIMESTAMP_FIELDS
            },
            "source_system": record.get("source_system"),
            "source_record_id": record.get("source_record_id"),
            "source_provenance": {
                field: record.get(field) for field in LOG_PROVENANCE_FIELDS
            },
        })

    keys = sorted(sensor_by_key)
    sensor_vectors, text_vectors, targets, has_logs, splits = [], [], [], [], []
    log_ids_by_window, entities_by_window = [], []
    severity_by_log_by_window, timestamps_by_log_by_window = [], []
    raw_timestamps_by_log_by_window = []
    source_ids_by_window, sensor_provenance_by_window, window_severity = [], [], []
    for key in keys:
        sensor = sensor_by_key[key]
        logs = sorted(logs_by_key[key], key=lambda item: item["log_id"])
        if not logs and missing_log_policy == "error":
            raise ValueError(f"Missing technician log for sensor window {key}.")
        if len(logs) > 1 and multiple_log_policy == "error":
            raise ValueError(f"Multiple technician logs for sensor window {key}.")
        sensor_vectors.append(sensor["embedding"])
        targets.append(sensor["rul"])
        splits.append(sensor["split"])
        sensor_provenance_by_window.append(sensor["provenance"])
        log_ids_by_window.append([log["log_id"] for log in logs])
        entities_by_window.append([entity for log in logs for entity in log["entities"]])
        severity_by_log_by_window.append([log["severity"] for log in logs])
        timestamps_by_log_by_window.append([log["timestamps"] for log in logs])
        raw_timestamps_by_log_by_window.append(
            [log["raw_timestamps"] for log in logs]
        )
        source_ids_by_window.append([
            {
                "log_id": log["log_id"],
                "source_system": log["source_system"],
                "source_record_id": log["source_record_id"],
                **log["source_provenance"],
            }
            for log in logs
        ])
        scores = [log["severity"]["severity_score"] for log in logs if log["severity"]["severity_score"] is not None]
        window_severity.append(max(scores) if scores else None)
        if logs:
            text_vectors.append(torch.stack([log["embedding"] for log in logs]).mean(0))
            has_logs.append(True)
        else:
            text_vectors.append(torch.zeros(768, dtype=torch.float32))
            has_logs.append(False)

    return {
        "namespace": namespace,
        "excluded_logs": excluded_logs,
        "unit_id": torch.tensor([key[0] for key in keys]),
        "cycle": torch.tensor([key[1] for key in keys]),
        "sensor_embeddings": torch.stack(sensor_vectors),
        "text_embeddings": torch.stack(text_vectors),
        "rul": torch.tensor(targets, dtype=torch.float32),
        "has_log": torch.tensor(has_logs, dtype=torch.bool),
        "split": tuple(splits),
        "log_ids_by_window": log_ids_by_window,
        "entities_by_window": entities_by_window,
        "severity_by_log_by_window": severity_by_log_by_window,
        "timestamps_by_log_by_window": timestamps_by_log_by_window,
        "raw_timestamps_by_log_by_window": raw_timestamps_by_log_by_window,
        "source_ids_by_window": source_ids_by_window,
        "sensor_provenance_by_window": sensor_provenance_by_window,
        "window_severity": window_severity,
    }
