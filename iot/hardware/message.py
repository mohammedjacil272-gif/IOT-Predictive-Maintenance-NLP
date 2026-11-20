"""Versioned, independent contract for live hardware telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
from typing import Mapping

HARDWARE_SCHEMA_VERSION = "1.0"
OPERATING_STATUSES = frozenset({"normal", "warming", "abnormal", "failure", "recovery"})
REQUIRED_FIELDS = frozenset({
    "schema_version", "device_id", "session_id", "timestamp", "sequence_number",
    "temperature", "vibration", "rpm", "pressure", "operating_status",
})


class HardwareTelemetryValidationError(ValueError):
    """Raised when hardware telemetry does not match schema 1.0."""


@dataclass(frozen=True)
class HardwareTelemetry:
    schema_version: str
    device_id: str
    session_id: str
    timestamp: datetime
    sequence_number: int
    temperature: float
    vibration: float
    rpm: int
    pressure: float
    operating_status: str

    def to_dict(self) -> dict[str, str | int | float]:
        """Return the canonical payload, with UTC time represented by `Z`."""
        timestamp = self.timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return {
            "schema_version": self.schema_version,
            "device_id": self.device_id,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "sequence_number": self.sequence_number,
            "temperature": self.temperature,
            "vibration": self.vibration,
            "rpm": self.rpm,
            "pressure": self.pressure,
            "operating_status": self.operating_status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)


def validate_payload(payload: Mapping[str, object]) -> HardwareTelemetry:
    if not isinstance(payload, Mapping):
        raise HardwareTelemetryValidationError("Telemetry payload must be a JSON object.")
    keys = set(payload)
    missing = sorted(REQUIRED_FIELDS - keys)
    extra = sorted(keys - REQUIRED_FIELDS)
    if missing:
        raise HardwareTelemetryValidationError(f"Missing required fields: {missing}")
    if extra:
        raise HardwareTelemetryValidationError(f"Unexpected fields: {extra}")
    if payload["schema_version"] != HARDWARE_SCHEMA_VERSION:
        raise HardwareTelemetryValidationError(f"schema_version must be {HARDWARE_SCHEMA_VERSION!r}.")
    device_id = _identifier(payload["device_id"], "device_id")
    session_id = _identifier(payload["session_id"], "session_id")
    timestamp = _utc_timestamp(payload["timestamp"])
    sequence = payload["sequence_number"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not 1 <= sequence <= 2**31 - 1:
        raise HardwareTelemetryValidationError("sequence_number must be an integer from 1 to 2147483647.")
    temperature = _number(payload["temperature"], "temperature", -40.0, 150.0)
    vibration = _number(payload["vibration"], "vibration", 0.0, 50.0)
    rpm = payload["rpm"]
    if isinstance(rpm, bool) or not isinstance(rpm, int) or not 0 <= rpm <= 10000:
        raise HardwareTelemetryValidationError("rpm must be an integer from 0 to 10000.")
    pressure = _number(payload["pressure"], "pressure", 0.0, 20.0)
    status = payload["operating_status"]
    if not isinstance(status, str) or status not in OPERATING_STATUSES:
        raise HardwareTelemetryValidationError(f"operating_status must be one of {sorted(OPERATING_STATUSES)}.")
    return HardwareTelemetry(HARDWARE_SCHEMA_VERSION, device_id, session_id, timestamp, sequence,
                             temperature, vibration, rpm, pressure, status)


def deserialize_payload(raw: str | bytes | bytearray) -> HardwareTelemetry:
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise HardwareTelemetryValidationError("Telemetry payload must be valid UTF-8 JSON.") from exc
    return validate_payload(decoded)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise HardwareTelemetryValidationError(f"{field} must be non-empty text of at most 64 characters.")
    if any(ord(char) < 33 for char in value):
        raise HardwareTelemetryValidationError(f"{field} must not contain whitespace or control characters.")
    return value


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise HardwareTelemetryValidationError("timestamp must be an ISO 8601 UTC string.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HardwareTelemetryValidationError("timestamp must be a valid ISO 8601 UTC string.") from exc
    if result.tzinfo is None or result.utcoffset() is None or result.utcoffset().total_seconds() != 0:
        raise HardwareTelemetryValidationError("timestamp must include a UTC timezone (Z or +00:00).")
    return result.astimezone(timezone.utc)


def _number(value: object, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HardwareTelemetryValidationError(f"{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise HardwareTelemetryValidationError(f"{field} must be finite and between {minimum:g} and {maximum:g}.")
    return result
