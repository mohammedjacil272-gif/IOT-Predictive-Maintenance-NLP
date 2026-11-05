"""Strict schema and timestamp validation for the Azure PdM CSV contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "PdM_telemetry.csv": ("datetime", "machineID", "volt", "rotate", "pressure", "vibration"),
    "PdM_errors.csv": ("datetime", "machineID", "errorID"),
    "PdM_maint.csv": ("datetime", "machineID", "comp"),
    "PdM_failures.csv": ("datetime", "machineID", "failure"),
    "PdM_machines.csv": ("machineID", "model", "age"),
}

EVENT_TABLES = frozenset({"PdM_errors.csv", "PdM_maint.csv", "PdM_failures.csv"})
APPROVED_SYNTHETIC_NAMESPACE = "azure_pdm_synthetic_fixture"
OPTIONAL_COLUMNS: dict[str, tuple[str, ...]] = {
    filename: ("available_at", "source_record_id") for filename in TABLE_SCHEMAS
}


class AzurePdMValidationError(ValueError):
    """Raised when a supplied Azure PdM value violates the contract."""


@dataclass(frozen=True)
class TimestampPolicy:
    """Explicit timestamp policy; naive timestamps are never assumed silently."""

    timezone_name: str
    allow_naive: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.timezone_name, str) or not self.timezone_name.strip():
            raise AzurePdMValidationError("timezone_name is required explicitly")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise AzurePdMValidationError(
                f"Unknown IANA timezone: {self.timezone_name!r}"
            ) from exc
        if not isinstance(self.allow_naive, bool):
            raise AzurePdMValidationError("allow_naive must be boolean")

    def parse(self, value: object, *, field: str = "datetime") -> datetime:
        if not isinstance(value, str) or not value.strip():
            raise AzurePdMValidationError(f"{field} must be a non-empty timestamp string")
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise AzurePdMValidationError(f"Malformed {field}: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            if not self.allow_naive:
                raise AzurePdMValidationError(
                    f"Naive {field} rejected; provide an explicit timezone policy"
                )
            parsed = parsed.replace(tzinfo=ZoneInfo(self.timezone_name))
        return parsed.astimezone(timezone.utc)

    def describe(self) -> dict[str, object]:
        return {
            "timezone_name": self.timezone_name,
            "allow_naive": self.allow_naive,
            "normalization": "UTC",
        }


def _fallback_source_record_id(filename: str, raw: Mapping[str, object]) -> str:
    canonical = json.dumps(
        {"source_file": filename, "fields": {key: str(raw.get(key, "")).strip() for key in sorted(raw)}},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"derived:{filename}:{hashlib.sha256(canonical).hexdigest()}"


def read_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV without coercing values or dropping malformed rows."""
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AzurePdMValidationError(f"Unable to read {path}: {exc}") from exc
    return rows, columns


def _required_text(value: object, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AzurePdMValidationError(f"{context}: {field} must be non-empty")
    return value.strip()


def _positive_int(value: object, field: str, context: str) -> int:
    text = _required_text(value, field, context)
    try:
        result = int(text)
    except (TypeError, ValueError) as exc:
        raise AzurePdMValidationError(f"{context}: {field} must be an integer") from exc
    if str(result) != text or result <= 0:
        raise AzurePdMValidationError(f"{context}: {field} must be a positive integer")
    return result


def _finite_float(value: object, field: str, context: str) -> float:
    text = _required_text(value, field, context)
    try:
        result = float(text)
    except (TypeError, ValueError) as exc:
        raise AzurePdMValidationError(f"{context}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise AzurePdMValidationError(f"{context}: {field} must be finite")
    return result


def normalize_rows(
    filename: str,
    rows: Iterable[Mapping[str, str]],
    *,
    timestamp_policy: TimestampPolicy,
) -> list[dict[str, Any]]:
    """Validate and normalize one known table, preserving source-row identity."""
    if filename not in TABLE_SCHEMAS:
        raise AzurePdMValidationError(f"Unsupported Azure PdM table: {filename}")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for row_number, raw in enumerate(rows, start=2):
        context = f"{filename} row {row_number}"
        if any(value is None or not str(value).strip() for value in raw.values()):
            raise AzurePdMValidationError(f"{context}: null or blank field")
        owner_source_id = raw.get("source_record_id")
        if owner_source_id is not None and str(owner_source_id).strip():
            source_record_id = str(owner_source_id).strip()
            source_record_id_kind = "owner_provided"
        else:
            source_record_id = _fallback_source_record_id(filename, raw)
            source_record_id_kind = "derived_content_hash"
        record: dict[str, Any] = {
            "source_file": filename,
            "source_record_id": source_record_id,
            "source_record_id_kind": source_record_id_kind,
        }
        if filename != "PdM_machines.csv":
            parsed_time = timestamp_policy.parse(raw.get("datetime"), field=f"{context}.datetime")
            if filename in EVENT_TABLES:
                record["event_time"] = parsed_time
                if raw.get("available_at") is not None and str(raw.get("available_at")).strip():
                    record["available_at"] = timestamp_policy.parse(
                        raw.get("available_at"), field=f"{context}.available_at"
                    )
            else:
                record["datetime"] = parsed_time
        record["machineID"] = _positive_int(raw.get("machineID"), "machineID", context)
        if filename == "PdM_telemetry.csv":
            for field in ("volt", "rotate", "pressure", "vibration"):
                record[field] = _finite_float(raw.get(field), field, context)
        elif filename == "PdM_machines.csv":
            record["model"] = _required_text(raw.get("model"), "model", context)
            record["age"] = _finite_float(raw.get("age"), "age", context)
            if record["age"] < 0:
                raise AzurePdMValidationError(f"{context}: age must be non-negative")
        else:
            field = {"PdM_errors.csv": "errorID", "PdM_maint.csv": "comp", "PdM_failures.csv": "failure"}[filename]
            record[field] = _required_text(raw.get(field), field, context)

        if filename == "PdM_machines.csv":
            duplicate_key = (record["machineID"],)
        elif filename == "PdM_telemetry.csv":
            duplicate_key = (record["machineID"], record["datetime"])
        else:
            event_field = {"PdM_errors.csv": "errorID", "PdM_maint.csv": "comp", "PdM_failures.csv": "failure"}[filename]
            duplicate_key = (record["machineID"], record["event_time"], record[event_field])
        if duplicate_key in seen:
            raise AzurePdMValidationError(f"{context}: duplicate record key {duplicate_key!r}")
        seen.add(duplicate_key)
        if any(existing["source_record_id"] == source_record_id for existing in normalized):
            raise AzurePdMValidationError(f"{context}: duplicate source_record_id {source_record_id!r}")
        normalized.append(record)
    return normalized
