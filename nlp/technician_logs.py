"""Fail-closed intake validation for permissioned technician-log data.

This module validates supplied records but never infers mappings, invents
timestamps, drops invalid rows, or generates embeddings. It is deliberately
separate from the checked-in synthetic fixture builder.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from nlp.preprocessing import normalize_text


REQUIRED_FIELDS = frozenset({
    "log_id", "source_system", "source_machine_id", "canonical_unit_id",
    "cycle", "event_timestamp", "created_at", "ingested_at", "available_at",
    "text", "event_type", "source_record_id", "source_dataset",
    "source_dataset_version", "provenance_reference", "is_synthetic",
    "approval_status",
})
TIMESTAMP_FIELDS = ("event_timestamp", "created_at", "ingested_at", "available_at")


@dataclass(frozen=True)
class GenuineLogValidationConfig:
    """Owner-supplied validation choices; no production defaults are inferred."""

    expected_dataset: str
    expected_dataset_version: str
    approved_status: str = "approved"
    allow_empty_text: bool = True

    def __post_init__(self) -> None:
        for name in ("expected_dataset", "expected_dataset_version", "approved_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")


def _text(value: Any, field: str, row_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {row_number}: {field} must be a non-empty string.")
    return value.strip()


def _positive_int(value: Any, field: str, row_number: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number}: {field} must be a positive integer.") from exc
    if parsed <= 0 or str(value).strip() != str(parsed):
        raise ValueError(f"Row {row_number}: {field} must be a positive integer.")
    return parsed


def _aware_timestamp(value: Any, field: str, row_number: int) -> str:
    text = _text(value, field, row_number).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field} must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Row {row_number}: {field} must include an explicit timezone offset.")
    return text


def _boolean_false(value: Any, field: str, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"false", "0", "no"}:
        raise ValueError(f"Row {row_number}: {field} must be false for genuine intake.")
    return False


def validate_genuine_logs(
    records: Iterable[Mapping[str, Any]],
    config: GenuineLogValidationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and deterministically normalize genuine log rows.

    All input rows are examined before returning. Invalid rows raise with their
    row number; no row is silently dropped. Multiple records for one
    (canonical_unit_id, cycle) are retained in log_id order for the fusion
    layer's explicit aggregation policy.
    """
    rows = list(records)
    if not rows:
        raise ValueError("Genuine log input contains no rows.")
    normalized: list[dict[str, Any]] = []
    seen_log_ids: set[str] = set()
    seen_source_ids: set[tuple[str, str]] = set()
    for row_number, source in enumerate(rows, 2):
        if not isinstance(source, Mapping):
            raise ValueError(f"Row {row_number}: record must be a mapping.")
        missing = REQUIRED_FIELDS - set(source)
        if missing:
            raise ValueError(f"Row {row_number}: missing required fields: {sorted(missing)}")
        record = dict(source)
        log_id = _text(record["log_id"], "log_id", row_number)
        if log_id in seen_log_ids:
            raise ValueError(f"Row {row_number}: duplicate log_id {log_id!r}.")
        seen_log_ids.add(log_id)
        source_system = _text(record["source_system"], "source_system", row_number)
        source_record_id = _text(record["source_record_id"], "source_record_id", row_number)
        source_key = (source_system, source_record_id)
        if source_key in seen_source_ids:
            raise ValueError(f"Row {row_number}: duplicate source record {source_key!r}.")
        seen_source_ids.add(source_key)
        dataset = _text(record["source_dataset"], "source_dataset", row_number)
        version = _text(record["source_dataset_version"], "source_dataset_version", row_number)
        if dataset != config.expected_dataset or version != config.expected_dataset_version:
            raise ValueError(f"Row {row_number}: source dataset/version does not match the approved intake config.")
        if _text(record["approval_status"], "approval_status", row_number) != config.approved_status:
            raise ValueError(f"Row {row_number}: approval_status is not approved for intake.")
        _boolean_false(record["is_synthetic"], "is_synthetic", row_number)
        record["log_id"] = log_id
        record["source_system"] = source_system
        record["source_record_id"] = source_record_id
        record["source_machine_id"] = _text(record["source_machine_id"], "source_machine_id", row_number)
        record["canonical_unit_id"] = _positive_int(record["canonical_unit_id"], "canonical_unit_id", row_number)
        record["cycle"] = _positive_int(record["cycle"], "cycle", row_number)
        record["provenance_reference"] = _text(record["provenance_reference"], "provenance_reference", row_number)
        record["event_type"] = _text(record["event_type"], "event_type", row_number)
        record["text"] = record["text"] if isinstance(record["text"], str) else ""
        if not config.allow_empty_text and not record["text"].strip():
            raise ValueError(f"Row {row_number}: text is empty and allow_empty_text is false.")
        record["normalized_text"] = normalize_text(record["text"]) if record["text"].strip() else ""
        record["text_status"] = "ok" if record["normalized_text"] else "empty_text"
        for field in TIMESTAMP_FIELDS:
            record[field] = _aware_timestamp(record[field], field, row_number)
        normalized.append(record)

    normalized.sort(key=lambda item: item["log_id"])
    counts: dict[tuple[int, int], int] = {}
    for row in normalized:
        key = (row["canonical_unit_id"], row["cycle"])
        counts[key] = counts.get(key, 0) + 1
    report = {
        "status": "PASS",
        "dataset": config.expected_dataset,
        "dataset_version": config.expected_dataset_version,
        "rows": len(normalized),
        "empty_text_rows": sum(row["text_status"] == "empty_text" for row in normalized),
        "multiple_log_windows": sum(count > 1 for count in counts.values()),
        "deterministic_order": [row["log_id"] for row in normalized],
        "dropped_rows": 0,
    }
    return normalized, report


def load_genuine_csv(
    path: str | Path, config: GenuineLogValidationConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a CSV and return validated rows plus a reproducible source hash."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Genuine log source is not a regular file: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Genuine log CSV has no header.")
        records = list(reader)
        columns = list(reader.fieldnames)
    prepared, report = validate_genuine_logs(records, config)
    report = {
        **report,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_columns": columns,
    }
    return prepared, report
