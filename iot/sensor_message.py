"""Validate local FD001 sensor messages before later IoT integration.

This module is intentionally independent of MQTT, InfluxDB, NLP, fusion, model
training, and preprocessing execution. It validates one C-MAPSS FD001 cycle
reading using the 17-feature order frozen by the existing preprocessing
configuration in ``preprocessing/cmapss_fd001.py``.

The C-MAPSS ``cycle`` field is an operating-cycle index, not a wall-clock
timestamp. Real-world timestamps should remain a separate concern in future
integration layers and are not accepted by this schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping


FD001_SENSOR_FEATURES: tuple[str, ...] = (
    "setting_1",
    "setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
)

_IDENTIFIER_FIELDS = ("unit_id", "cycle")
_REQUIRED_KEYS = set(_IDENTIFIER_FIELDS) | set(FD001_SENSOR_FEATURES)


class SensorMessageValidationError(ValueError):
    """Raised when a local FD001 sensor message does not match the schema."""


@dataclass(frozen=True)
class SensorMessage:
    """Normalized FD001 sensor message.

    Attributes:
        unit_id: Positive integer C-MAPSS engine/unit identifier.
        cycle: Positive integer C-MAPSS operating-cycle index.
        features: Finite numeric feature values keyed by the exact 17 selected
            FD001 feature names used by the frozen sensor-only baseline.
    """

    unit_id: int
    cycle: int
    features: dict[str, float]

    def to_dict(self) -> dict[str, int | float]:
        """Return a flat message dictionary in the documented feature order."""

        return {
            "unit_id": self.unit_id,
            "cycle": self.cycle,
            **{feature: self.features[feature] for feature in FD001_SENSOR_FEATURES},
        }


def validate_sensor_message(message: Mapping[str, object]) -> SensorMessage:
    """Validate and normalize a flat local FD001 sensor message.

    Args:
        message: Mapping containing ``unit_id``, ``cycle``, and exactly the 17
            feature names in ``FD001_SENSOR_FEATURES``.

    Returns:
        A ``SensorMessage`` with strict integer identifiers and feature values
        normalized to ``float``.

    Raises:
        SensorMessageValidationError: If required fields are missing, extra
            fields are present, identifiers are not positive integers, or any
            feature value is not finite numeric data.
    """

    if not isinstance(message, Mapping):
        raise SensorMessageValidationError("Sensor message must be a mapping.")

    keys = set(message.keys())
    missing = sorted(_REQUIRED_KEYS - keys)
    if missing:
        raise SensorMessageValidationError(f"Missing required fields: {missing}")

    unexpected = sorted(keys - _REQUIRED_KEYS)
    if unexpected:
        raise SensorMessageValidationError(f"Unexpected fields: {unexpected}")

    unit_id = _validate_positive_int(message["unit_id"], "unit_id")
    cycle = _validate_positive_int(message["cycle"], "cycle")
    features = {
        feature: _validate_feature_value(message[feature], feature)
        for feature in FD001_SENSOR_FEATURES
    }

    return SensorMessage(unit_id=unit_id, cycle=cycle, features=features)


def _validate_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SensorMessageValidationError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise SensorMessageValidationError(f"{field_name} must be greater than 0.")
    return value


def _validate_feature_value(value: object, feature_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SensorMessageValidationError(f"{feature_name} must be a finite numeric value.")

    normalized = float(value)
    if not math.isfinite(normalized):
        raise SensorMessageValidationError(f"{feature_name} must be finite.")

    return normalized
