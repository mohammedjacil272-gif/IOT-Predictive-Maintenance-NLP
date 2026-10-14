
"""Metadata-based alignment for multimodal FD001 samples."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def build_fusion_samples(
    sensor_records: Sequence[Mapping],
    text_records: Sequence[Mapping],
) -> dict[str, Any]:
    """Align sensor windows and NLP records by (unit_id, cycle).

    Sensor records require:
        unit_id, cycle, split, sensor_embedding, rul

    Text records require:
        log_id, unit_id, cycle, split, text_embedding

    Optional text-record provenance:
        entities: sequence of entity dictionaries
        severity_score: numeric severity score
        severity_label: severity label
        matched_terms: matched severity terms
        explanation: severity explanation

    Missing text produces a zero vector and has_log=False.
    Multiple logs are averaged in deterministic log_id order.
    Per-log provenance is retained separately from the averaged embedding.
    """

    sensor_by_key = {}

    # Validate and index sensor windows.
    for record in sensor_records:
        key = (record["unit_id"], record["cycle"])

        if key in sensor_by_key:
            raise ValueError(f"Duplicate sensor window: {key}")

        embedding = torch.as_tensor(
            record["sensor_embedding"], dtype=torch.float32
        )
        if embedding.shape != (64,):
            raise ValueError(
                f"Sensor embedding for {key} must have 64 values."
            )
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite sensor embedding for {key}.")

        rul = float(record["rul"])
        if not torch.isfinite(torch.tensor(rul)):
            raise ValueError(f"Non-finite RUL target for {key}.")

        sensor_by_key[key] = {
            "embedding": embedding,
            "rul": rul,
            "split": record["split"],
        }

    if not sensor_by_key:
        raise ValueError("No sensor records supplied.")

    text_by_key = defaultdict(list)
    seen_log_ids = set()

    # Validate and group NLP records by sensor window.
    for record in text_records:
        key = (record["unit_id"], record["cycle"])

        if key not in sensor_by_key:
            raise ValueError(
                f"Text record has no matching sensor window: {key}"
            )

        if record["split"] != sensor_by_key[key]["split"]:
            raise ValueError(f"Split mismatch for {key}")

        log_id = str(record["log_id"]).strip()
        if not log_id:
            raise ValueError("Text record has a missing or empty log_id.")

        if log_id in seen_log_ids:
            raise ValueError(f"Duplicate log_id: {log_id}")

        seen_log_ids.add(log_id)

        embedding = torch.as_tensor(
            record["text_embedding"], dtype=torch.float32
        )
        if embedding.shape != (768,):
            raise ValueError(
                f"Text embedding for {key} must have 768 values."
            )
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite text embedding for {key}.")

        # Preserve entities and validate their source-log provenance.
        raw_entities = record.get("entities", [])
        if raw_entities is None:
            raw_entities = []

        if not isinstance(raw_entities, Sequence) or isinstance(
            raw_entities, (str, bytes)
        ):
            raise ValueError(
                f"Entities for log {log_id} must be a sequence."
            )

        entities = []
        expected_metadata = {
            "log_id": log_id,
            "unit_id": record["unit_id"],
            "cycle": record["cycle"],
            "split": record["split"],
        }

        for entity in raw_entities:
            if not isinstance(entity, Mapping):
                raise ValueError(
                    f"Each entity for log {log_id} must be a mapping."
                )

            entity_copy = dict(entity)

            for field, expected_value in expected_metadata.items():
                if (
                    field in entity_copy
                    and entity_copy[field] != expected_value
                ):
                    raise ValueError(
                        f"Entity provenance mismatch for log {log_id}: "
                        f"{field}"
                    )

                entity_copy[field] = expected_value

            entities.append(entity_copy)

        # Severity is optional for compatibility with embedding-only callers.
        severity_score = record.get("severity_score")
        if severity_score is not None:
            severity_score = float(severity_score)
            if not torch.isfinite(torch.tensor(severity_score)):
                raise ValueError(
                    f"Non-finite severity score for log {log_id}."
                )

        severity = {
            "log_id": log_id,
            "unit_id": record["unit_id"],
            "cycle": record["cycle"],
            "split": record["split"],
            "severity_score": severity_score,
            "severity_label": record.get("severity_label"),
            "matched_terms": record.get("matched_terms"),
            "explanation": record.get("explanation"),
        }

        text_by_key[key].append(
            {
                "log_id": log_id,
                "embedding": embedding,
                "entities": entities,
                "severity": severity,
            }
        )

    # Sort windows and logs for deterministic output.
    keys = sorted(sensor_by_key)

    sensor_vectors = []
    text_vectors = []
    targets = []
    has_logs = []
    splits = []

    log_ids_by_window = []
    entities_by_window = []
    severity_by_log_by_window = []
    window_severity = []

    for key in keys:
        sensor = sensor_by_key[key]
        logs = sorted(
            text_by_key[key],
            key=lambda item: item["log_id"],
        )

        sensor_vectors.append(sensor["embedding"])
        targets.append(sensor["rul"])
        splits.append(sensor["split"])

        log_ids_by_window.append(
            [log["log_id"] for log in logs]
        )

        window_entities = []
        window_severities = []

        for log in logs:
            window_entities.extend(log["entities"])
            window_severities.append(log["severity"])

        entities_by_window.append(window_entities)
        severity_by_log_by_window.append(window_severities)

        available_scores = [
            item["severity_score"]
            for item in window_severities
            if item["severity_score"] is not None
        ]

        window_severity.append(
            max(available_scores) if available_scores else None
        )

        if logs:
            text_vectors.append(
                torch.stack(
                    [log["embedding"] for log in logs]
                ).mean(dim=0)
            )
            has_logs.append(True)
        else:
            text_vectors.append(
                torch.zeros(768, dtype=torch.float32)
            )
            has_logs.append(False)

    return {
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
        "window_severity": window_severity,
    }
