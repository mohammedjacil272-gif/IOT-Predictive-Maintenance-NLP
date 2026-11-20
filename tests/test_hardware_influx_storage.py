import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from iot.hardware.influx_storage import (
    DEFAULT_HARDWARE_MEASUREMENT, HardwareInfluxError, InfluxStorageConfig, build_point, write_telemetry,
)
from iot.hardware.message import validate_payload
from tests.test_hardware_message import payload


class FakePoint:
    def __init__(self, measurement): self.measurement, self.tags, self.fields, self.timestamp = measurement, {}, {}, None
    def tag(self, key, value): self.tags[key] = value; return self
    def field(self, key, value): self.fields[key] = value; return self
    def time(self, value): self.timestamp = value; return self


class TestHardwareInfluxStorage(unittest.TestCase):
    def test_point_uses_dedicated_measurement_fields_tags_and_event_time(self):
        fake_module = SimpleNamespace(Point=FakePoint)
        with patch("iot.hardware.influx_storage.importlib.import_module", return_value=fake_module):
            telemetry = validate_payload(payload())
            point = build_point(telemetry)
        self.assertEqual(point.measurement, DEFAULT_HARDWARE_MEASUREMENT)
        self.assertEqual(point.tags, {"device_id": "ESP32_001", "session_id": "session-a"})
        self.assertEqual(point.fields["sequence_number"], 1)
        self.assertEqual(point.fields["operating_status"], "normal")
        self.assertEqual(point.timestamp, datetime(2026, 11, 20, 10, 30, tzinfo=timezone.utc))

    def test_write_api_receives_point_and_configured_bucket(self):
        fake_module = SimpleNamespace(Point=FakePoint)
        seen = []
        class Write:
            def write(self, **kwargs): seen.append(kwargs)
        config = InfluxStorageConfig("http://localhost:8086", "IOT", "hardware", "token")
        with patch("iot.hardware.influx_storage.importlib.import_module", return_value=fake_module):
            write_telemetry(validate_payload(payload()), config, Write())
        self.assertEqual(seen[0]["bucket"], "hardware")

    def test_invalid_config_rejected(self):
        with self.assertRaisesRegex(HardwareInfluxError, "url is required"):
            InfluxStorageConfig("", "IOT", "bucket", "token").validate()


if __name__ == "__main__":
    unittest.main()
