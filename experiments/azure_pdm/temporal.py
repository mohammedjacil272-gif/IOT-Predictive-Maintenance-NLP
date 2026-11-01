"""Fail-closed temporal validation for Azure PdM event records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .schema import AzurePdMValidationError, TimestampPolicy


class FutureEventError(AzurePdMValidationError):
    """Raised when an event is after the prediction cutoff."""


def parse_cutoff(value: object, *, timestamp_policy: TimestampPolicy) -> datetime:
    return timestamp_policy.parse(value, field="prediction_cutoff")


def _utc_value(value: object, *, field: str, timestamp_policy: TimestampPolicy) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise AzurePdMValidationError(f"Naive {field} rejected")
        return value.astimezone(timezone.utc)
    return timestamp_policy.parse(value, field=field)


def validate_event_availability(
    records: Iterable[Mapping[str, Any]],
    *,
    timestamp_policy: TimestampPolicy,
    require_availability: bool = True,
) -> list[Mapping[str, Any]]:
    """Require separate event and availability timestamps; never substitute one for the other."""
    validated: list[Mapping[str, Any]] = []
    for record in records:
        event_time = _utc_value(record.get("event_time"), field="event_time", timestamp_policy=timestamp_policy)
        available_value = record.get("available_at")
        if available_value is None or not str(available_value).strip():
            if require_availability:
                raise AzurePdMValidationError(
                    f"Missing available_at for source record {record.get('source_record_id', 'unknown')!r}"
                )
            validated.append(record)
            continue
        available_at = _utc_value(available_value, field="available_at", timestamp_policy=timestamp_policy)
        if available_at < event_time:
            raise AzurePdMValidationError(
                f"available_at precedes event_time for source record {record.get('source_record_id', 'unknown')!r}"
            )
        validated.append(record)
    return validated


def reject_future_events(
    records: Iterable[Mapping[str, Any]],
    cutoff: object,
    *,
    timestamp_policy: TimestampPolicy,
    timestamp_field: str = "event_time",
) -> list[Mapping[str, Any]]:
    """Reject future event or availability times; do not silently filter records."""
    cutoff_utc = parse_cutoff(cutoff, timestamp_policy=timestamp_policy)
    validated = validate_event_availability(
        records, timestamp_policy=timestamp_policy, require_availability=True
    )
    eligible: list[Mapping[str, Any]] = []
    for record in validated:
        value = record.get(timestamp_field)
        event_time_utc = _utc_value(value, field=timestamp_field, timestamp_policy=timestamp_policy)
        if event_time_utc > cutoff_utc:
            source_id = record.get("source_record_id", "unknown")
            raise FutureEventError(
                f"Future event {source_id!r} at {event_time_utc.isoformat()} exceeds cutoff {cutoff_utc.isoformat()}"
            )
        available_at_utc = _utc_value(
            record.get("available_at"), field="available_at", timestamp_policy=timestamp_policy
        )
        if available_at_utc > cutoff_utc:
            source_id = record.get("source_record_id", "unknown")
            raise FutureEventError(
                f"Future availability for {source_id!r} at {available_at_utc.isoformat()} exceeds cutoff {cutoff_utc.isoformat()}"
            )
        eligible.append(record)
    return eligible
