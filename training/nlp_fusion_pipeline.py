"""Enforced synthetic NLP-artifact to fusion orchestration boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from training.fusion_dataset import build_fusion_samples
from training.nlp_artifact_validation import (
    ArtifactValidationLimits,
    load_validated_nlp_artifact_package,
)
from training.nlp_fusion_adapter import adapt_nlp_embedding_metadata


def build_validated_fusion_samples(
    package_dir: str,
    sensor_records: Sequence[Mapping[str, Any]],
    *,
    missing_log_policy: str,
    multiple_log_policy: str,
    limits: ArtifactValidationLimits | None = None,
    genuine_approval: Mapping | None = None,
    point_in_time_policy: Mapping | None = None,
    prediction_cutoff: object | None = None,
) -> dict[str, Any]:
    """Validate NLP artifacts before adapting or joining them with sensors.

    This is the only supported package-to-fusion entry point. The validated
    package loader fails before adapter invocation unless its report status is
    ``PASS``. Synthetic packages remain explicitly synthetic; this function
    does not infer or manufacture genuine provenance.
    """
    package = load_validated_nlp_artifact_package(package_dir, limits=limits)
    metadata = package["metadata"]
    processed_logs = package["processed_logs"]
    embeddings = package["embeddings"]
    source_report = package["verification_report"]
    validation_report = package["validation_report"]

    records = adapt_nlp_embedding_metadata(
        metadata,
        embeddings,
        processed_logs=processed_logs,
        verification_report=source_report,
    )
    _attach_validated_annotations(records, package["entities"], package["severity"])

    samples = build_fusion_samples(
        sensor_records,
        records,
        missing_log_policy=missing_log_policy,
        multiple_log_policy=multiple_log_policy,
        genuine_approval=genuine_approval,
        point_in_time_policy=point_in_time_policy,
        prediction_cutoff=prediction_cutoff,
    )
    return {
        "samples": samples,
        "validation": validation_report,
        "source_report": source_report,
        "namespace": samples["namespace"],
        "log_ids": samples["log_ids_by_window"],
    }


def _attach_validated_annotations(
    records: list[dict[str, Any]],
    entities: Sequence[Mapping[str, Any]],
    severity: Sequence[Mapping[str, str]],
) -> None:
    """Attach only records already linked to adapted log IDs."""
    by_log_id = {record["log_id"]: record for record in records}
    entities_by_log: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        log_id = entity.get("log_id")
        if not isinstance(log_id, str) or not log_id.strip():
            raise ValueError("Validated entity is missing a non-empty log_id.")
        if log_id not in by_log_id:
            raise ValueError(f"Validated entity references unknown log_id: {log_id}")
        entities_by_log.setdefault(log_id, []).append(dict(entity))

    severity_by_log: dict[str, dict[str, str]] = {}
    for item in severity:
        log_id = item.get("log_id", "").strip()
        if not log_id or log_id not in by_log_id:
            raise ValueError(f"Validated severity references unknown log_id: {log_id!r}")
        if log_id in severity_by_log:
            raise ValueError(f"Duplicate validated severity log_id: {log_id}")
        severity_by_log[log_id] = dict(item)

    for record in records:
        log_id = record["log_id"]
        record["entities"] = entities_by_log.get(log_id, [])
        severity_item = severity_by_log.get(log_id)
        if severity_item is not None:
            for field in ("severity_score", "severity_label", "matched_terms", "explanation"):
                if field in severity_item:
                    record[field] = severity_item[field]
