"""Deterministic, explicitly synthetic text derived from structured events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

from .schema import AzurePdMValidationError, TimestampPolicy
from .temporal import reject_future_events


DERIVED_NAMESPACE = "derived_synthetic"


def _event_time_text(value: object, policy: TimestampPolicy, *, field: str) -> str:
    parsed = value if isinstance(value, datetime) else policy.parse(value, field=field)
    return parsed.isoformat().replace("+00:00", "Z")


def _derived_record(
    record: Mapping[str, Any],
    *,
    event_type: str,
    value_field: str,
    timestamp_policy: TimestampPolicy,
) -> dict[str, Any]:
    source_file = str(record.get("source_file", "")).strip()
    source_id = str(record.get("source_record_id", "")).strip()
    if not source_file or not source_id:
        raise AzurePdMValidationError("Derived events require source_file and source_record_id")
    machine_id = record.get("machineID")
    if not isinstance(machine_id, int) or machine_id <= 0:
        raise AzurePdMValidationError("Derived events require a positive integer machineID")
    value = str(record.get(value_field, "")).strip()
    if not value:
        raise AzurePdMValidationError(f"Derived events require non-empty {value_field}")
    event_time = _event_time_text(record.get("event_time"), timestamp_policy, field="event_time")
    available_at = _event_time_text(record.get("available_at"), timestamp_policy, field="available_at")
    if event_type == "error":
        text = f"structured error event error_id={value} machine_id={machine_id} observed_at={event_time}"
    elif event_type == "maintenance":
        text = f"structured maintenance event component={value} machine_id={machine_id} observed_at={event_time}"
    else:
        raise AzurePdMValidationError(f"Unsupported derived event type: {event_type}")
    return {
        "namespace": DERIVED_NAMESPACE,
        "is_synthetic": True,
        "source_file": source_file,
        "source_record_id": source_id,
        "source_record_id_kind": str(record.get("source_record_id_kind", "")).strip(),
        "machineID": machine_id,
        "event_type": event_type,
        "source_field": value_field,
        "source_value": value,
        "event_time": event_time,
        "available_at": available_at,
        "text": text,
        "template_version": "azure_pdm_structured_event_v1",
    }


def derive_event_text_records(
    errors: Iterable[Mapping[str, Any]],
    maintenance: Iterable[Mapping[str, Any]],
    *,
    prediction_cutoff: object,
    timestamp_policy: TimestampPolicy,
) -> list[dict[str, Any]]:
    """Create deterministic synthetic text only from events available by cutoff."""
    eligible_errors = reject_future_events(
        errors, prediction_cutoff, timestamp_policy=timestamp_policy
    )
    eligible_maintenance = reject_future_events(
        maintenance, prediction_cutoff, timestamp_policy=timestamp_policy
    )
    derived = [
        _derived_record(
            record,
            event_type="error",
            value_field="errorID",
            timestamp_policy=timestamp_policy,
        )
        for record in eligible_errors
    ]
    derived.extend(
        _derived_record(
            record,
            event_type="maintenance",
            value_field="comp",
            timestamp_policy=timestamp_policy,
        )
        for record in eligible_maintenance
    )
    return sorted(
        derived,
        key=lambda record: (record["event_time"], record["machineID"], record["source_record_id"]),
    )


def validate_derived_synthetic_record(record: Mapping[str, Any]) -> None:
    if record.get("namespace") != DERIVED_NAMESPACE:
        raise AzurePdMValidationError("Derived event namespace must be derived_synthetic")
    if record.get("is_synthetic") is not True:
        raise AzurePdMValidationError("Derived event is_synthetic must be true")
    if record.get("template_version") != "azure_pdm_structured_event_v1":
        raise AzurePdMValidationError("Unrecognized derived event template_version")
    if not isinstance(record.get("machineID"), int) or record["machineID"] <= 0:
        raise AzurePdMValidationError("Derived event machineID must be a positive integer")
    expected_fields = {"error": ("PdM_errors.csv", "errorID"), "maintenance": ("PdM_maint.csv", "comp")}
    event_type = record.get("event_type")
    if event_type not in expected_fields:
        raise AzurePdMValidationError("Unsupported derived event type")
    expected_file, expected_field = expected_fields[event_type]
    if record.get("source_file") != expected_file or record.get("source_field") != expected_field:
        raise AzurePdMValidationError("Derived event source metadata is inconsistent")
    if not isinstance(record.get("source_record_id"), str) or not record["source_record_id"].strip():
        raise AzurePdMValidationError("Derived event requires non-empty source_record_id")
    if not isinstance(record.get("source_record_id_kind"), str) or record["source_record_id_kind"] not in {"owner_provided", "derived_content_hash"}:
        raise AzurePdMValidationError("Derived event source_record_id_kind is invalid")
    if record["source_record_id_kind"] == "derived_content_hash" and not record["source_record_id"].startswith(f"derived:{record['source_file']}:"):
        raise AzurePdMValidationError("Derived source_record_id does not match its source file")
    for field in ("source_value", "event_time", "available_at", "text"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise AzurePdMValidationError(f"Derived event requires non-empty {field}")
    policy = TimestampPolicy("UTC", allow_naive=False)
    event_time = _event_time_text(record["event_time"], policy, field="event_time")
    available_at = _event_time_text(record["available_at"], policy, field="available_at")
    if available_at < event_time:
        raise AzurePdMValidationError("Derived event available_at precedes event_time")
    value = record["source_value"]
    if event_type == "error":
        expected_text = f"structured error event error_id={value} machine_id={record['machineID']} observed_at={event_time}"
    else:
        expected_text = f"structured maintenance event component={value} machine_id={record['machineID']} observed_at={event_time}"
    if record["text"] != expected_text:
        raise AzurePdMValidationError("Derived event text does not match structured source fields")
