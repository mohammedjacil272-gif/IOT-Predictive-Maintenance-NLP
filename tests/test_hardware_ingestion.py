import unittest

from iot.hardware.message import validate_payload
from iot.hardware.influx_storage import InfluxStorageConfig
from iot.hardware.mqtt_ingestion import (
    HardwareMqttError, IngestionStats, MqttIngestionConfig, SequenceTracker,
    process_payload, start_ingestion,
)
from tests.test_hardware_message import payload


class TestHardwareIngestion(unittest.TestCase):
    def test_valid_payload_written_once_and_duplicate_or_old_sequence_skipped(self):
        tracker, stats, written = SequenceTracker(), IngestionStats(), []
        raw = validate_payload(payload()).to_json()
        self.assertTrue(process_payload(raw, tracker=tracker, stats=stats, writer=written.append))
        self.assertFalse(process_payload(raw, tracker=tracker, stats=stats, writer=written.append))
        self.assertEqual((stats.received, stats.accepted, stats.duplicates), (2, 1, 1))
        self.assertEqual(len(written), 1)

    def test_malformed_payload_and_storage_failure_do_not_escape_callback(self):
        stats, tracker = IngestionStats(), SequenceTracker()
        self.assertFalse(process_payload("{bad", tracker=tracker, stats=stats, writer=lambda _: None))
        raw = validate_payload(payload(sequence_number=2)).to_json()
        self.assertFalse(process_payload(raw, tracker=tracker, stats=stats, writer=lambda _: (_ for _ in ()).throw(RuntimeError("down"))))
        self.assertEqual((stats.invalid, stats.write_errors), (1, 1))

    def test_failed_storage_does_not_advance_deduplication_watermark(self):
        stats, tracker = IngestionStats(), SequenceTracker()
        raw = validate_payload(payload(sequence_number=3)).to_json()
        self.assertFalse(process_payload(raw, tracker=tracker, stats=stats,
                                         writer=lambda _: (_ for _ in ()).throw(RuntimeError("offline"))))
        self.assertTrue(process_payload(raw, tracker=tracker, stats=stats, writer=lambda _: None))
        self.assertEqual((stats.write_errors, stats.accepted), (1, 1))

    def test_deduplication_is_scoped_to_device_and_session(self):
        tracker = SequenceTracker()
        first = validate_payload(payload(sequence_number=1))
        second_session = validate_payload(payload(session_id="session-b", sequence_number=1))
        other_device = validate_payload(payload(device_id="ESP32_002", sequence_number=1))
        self.assertFalse(tracker.is_duplicate(first))
        tracker.mark_stored(first)
        self.assertTrue(tracker.is_duplicate(first))
        self.assertFalse(tracker.is_duplicate(second_session))
        self.assertFalse(tracker.is_duplicate(other_device))

    def test_environment_config_has_independent_topic_and_auth(self):
        config = MqttIngestionConfig.from_env({"IOT_HARDWARE_MQTT_USERNAME": "user", "IOT_HARDWARE_MQTT_PASSWORD": "pass"})
        self.assertEqual(config.topic, "maintainiq/hardware/telemetry")
        self.assertEqual(config.username, "user")

    def test_mqtt_subscriber_receives_valid_message_and_connection_error_is_reported(self):
        class Message: payload = validate_payload(payload()).to_json().encode()
        class Client:
            def connect(self, host, port, keepalive=60):
                self.connected_to = (host, port)
                self.on_connect(self, None, {}, 0, None)
            def subscribe(self, topic, qos=0): self.subscription = (topic, qos)
            def loop_forever(self): self.on_message(self, None, Message())
            def disconnect(self): self.disconnected = True
        client = Client()
        written = []
        mqtt = MqttIngestionConfig()
        influx = InfluxStorageConfig("http://localhost:8086", "IOT", "bucket", "token")
        stats = start_ingestion(mqtt, influx, client_factory=lambda: client, writer=written.append)
        self.assertEqual((stats.accepted, len(written)), (1, 1))
        self.assertEqual(client.subscription[0], "maintainiq/hardware/telemetry")
        self.assertTrue(client.disconnected)

        class FailingClient(Client):
            def connect(self, host, port, keepalive=60): raise OSError("offline")
        with self.assertRaises(HardwareMqttError):
            start_ingestion(mqtt, influx, client_factory=FailingClient, writer=written.append)


if __name__ == "__main__":
    unittest.main()
