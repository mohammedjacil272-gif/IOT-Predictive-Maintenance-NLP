import json
import unittest
from datetime import datetime, timezone

from iot.hardware.message import (
    HardwareTelemetryValidationError,
    deserialize_payload,
    validate_payload,
)


def payload(**overrides):
    value = {
        "schema_version": "1.0", "device_id": "ESP32_001", "session_id": "session-a",
        "timestamp": "2026-11-20T10:30:00Z", "sequence_number": 1,
        "temperature": 45.6, "vibration": 1.3, "rpm": 1500, "pressure": 3.2,
        "operating_status": "normal",
    }
    value.update(overrides)
    return value


class TestHardwareMessage(unittest.TestCase):
    def test_valid_payload_and_canonical_json(self):
        telemetry = validate_payload(payload())
        self.assertEqual(telemetry.timestamp, datetime(2026, 11, 20, 10, 30, tzinfo=timezone.utc))
        self.assertEqual(json.loads(telemetry.to_json())["timestamp"], "2026-11-20T10:30:00.000Z")
        self.assertEqual(deserialize_payload(telemetry.to_json()), telemetry)

    def test_missing_and_extra_fields_rejected(self):
        value = payload(); del value["pressure"]
        with self.assertRaisesRegex(HardwareTelemetryValidationError, "Missing required"):
            validate_payload(value)
        with self.assertRaisesRegex(HardwareTelemetryValidationError, "Unexpected fields"):
            validate_payload(payload(rul=3))

    def test_invalid_types_version_time_and_ranges_rejected(self):
        cases = (
            ({"schema_version": "2.0"}, "schema_version"),
            ({"timestamp": "2026-11-20T10:30:00"}, "UTC timezone"),
            ({"timestamp": "not-time"}, "ISO 8601"),
            ({"temperature": True}, "temperature must be numeric"),
            ({"temperature": 151}, "between -40 and 150"),
            ({"vibration": -0.1}, "between 0 and 50"),
            ({"rpm": 1.2}, "rpm must be an integer"),
            ({"pressure": float("nan")}, "finite"),
            ({"sequence_number": 0}, "sequence_number"),
            ({"operating_status": "unknown"}, "operating_status"),
        )
        for change, message in cases:
            with self.subTest(change=change), self.assertRaisesRegex(HardwareTelemetryValidationError, message):
                validate_payload(payload(**change))

    def test_malformed_json_is_normalized_to_contract_error(self):
        with self.assertRaisesRegex(HardwareTelemetryValidationError, "valid UTF-8 JSON"):
            deserialize_payload(b"{bad-json")


if __name__ == "__main__":
    unittest.main()
