"""Independent MaintainIQ hardware telemetry contract and adapters."""

from .message import (
    HARDWARE_SCHEMA_VERSION,
    HardwareTelemetry,
    HardwareTelemetryValidationError,
    deserialize_payload,
    validate_payload,
)

__all__ = [
    "HARDWARE_SCHEMA_VERSION",
    "HardwareTelemetry",
    "HardwareTelemetryValidationError",
    "deserialize_payload",
    "validate_payload",
]
