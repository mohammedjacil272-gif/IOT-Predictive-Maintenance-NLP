import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from iot.mqtt_publisher import (
    DEFAULT_MQTT_HOST,
    DEFAULT_MQTT_PORT,
    DEFAULT_MQTT_QOS,
    DEFAULT_MQTT_TOPIC,
    MqttPublisherConfig,
    MqttPublisherError,
    main,
    parse_config,
    publish_replay,
    serialize_sensor_message,
)
from iot.sensor_message import FD001_SENSOR_FEATURES


FIRST_FD001_ROW = (
    "1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 "
    "21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 "
    "8.4195 0.03 392 2388 100.00 39.06 23.4190"
)

SECOND_FD001_ROW = (
    "1 2 0.0019 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 "
    "21.61 553.75 2388.04 9044.07 1.30 47.49 522.28 2388.07 8131.49 "
    "8.4318 0.03 392 2388 100.00 39.00 23.4236"
)


class PublishResult:
    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.waited = False

    def wait_for_publish(self) -> None:
        self.waited = True


class MockMqttClient:
    def __init__(self, *, connect_rc: int = 0, publish_rc: int = 0, raise_on_publish: Exception | None = None) -> None:
        self.connect_rc = connect_rc
        self.publish_rc = publish_rc
        self.raise_on_publish = raise_on_publish
        self.connected_to: tuple[str, int] | None = None
        self.published: list[tuple[str, str, int]] = []
        self.disconnected = False

    def connect(self, host: str, port: int) -> int:
        self.connected_to = (host, port)
        return self.connect_rc

    def publish(self, topic: str, payload: str, qos: int) -> PublishResult:
        if self.raise_on_publish is not None:
            raise self.raise_on_publish
        self.published.append((topic, payload, qos))
        return PublishResult(self.publish_rc)

    def disconnect(self) -> int:
        self.disconnected = True
        return 0


def write_temp_fd001(directory: Path, contents: str) -> Path:
    path = directory / "train_FD001.txt"
    path.write_text(contents, encoding="utf-8")
    return path


class TestMqttPublisher(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp_directory.name)
        self.train_path = write_temp_fd001(self.temp_path, FIRST_FD001_ROW + "\n" + SECOND_FD001_ROW + "\n")

    def tearDown(self) -> None:
        self._temp_directory.cleanup()

    def test_parse_config_uses_defaults(self) -> None:
        config = parse_config([], env={})

        self.assertEqual(config.host, DEFAULT_MQTT_HOST)
        self.assertEqual(config.port, DEFAULT_MQTT_PORT)
        self.assertEqual(config.topic, DEFAULT_MQTT_TOPIC)
        self.assertEqual(config.qos, DEFAULT_MQTT_QOS)
        self.assertIsNone(config.limit)
        self.assertFalse(config.dry_run)

    def test_parse_config_uses_environment_and_cli_override(self) -> None:
        env = {
            "IOT_MQTT_HOST": "broker.local",
            "IOT_MQTT_PORT": "1884",
            "IOT_MQTT_TOPIC": "factory/line-a",
            "IOT_MQTT_QOS": "1",
            "IOT_REPLAY_LIMIT": "7",
            "IOT_TRAIN_PATH": "data/from-env.txt",
        }

        config = parse_config(["--host", "cli-broker", "--limit", "2", "--dry-run"], env=env)

        self.assertEqual(config.host, "cli-broker")
        self.assertEqual(config.port, 1884)
        self.assertEqual(config.topic, "factory/line-a")
        self.assertEqual(config.qos, 1)
        self.assertEqual(config.limit, 2)
        self.assertEqual(config.train_path, Path("data/from-env.txt"))
        self.assertTrue(config.dry_run)

    def test_invalid_qos_is_rejected(self) -> None:
        with self.assertRaisesRegex(MqttPublisherError, "QoS"):
            parse_config(["--qos", "3"], env={})

    def test_serialize_message_contains_only_sensor_schema_fields(self) -> None:
        config = MqttPublisherConfig(train_path=self.train_path, limit=1, dry_run=True)
        output = io.StringIO()

        publish_replay(config, output=output)
        payload = output.getvalue().split("payload=", maxsplit=1)[1].strip()
        decoded = json.loads(payload)

        self.assertEqual(decoded["unit_id"], 1)
        self.assertEqual(decoded["cycle"], 1)
        self.assertEqual([key for key in decoded if key not in ("unit_id", "cycle")], list(FD001_SENSOR_FEATURES))
        self.assertNotIn("rul", decoded)
        self.assertNotIn("failure_label", decoded)
        self.assertNotIn("timestamp", decoded)

    def test_publish_replay_uses_mock_client(self) -> None:
        client = MockMqttClient()
        config = MqttPublisherConfig(
            host="broker.local",
            port=1884,
            topic="industrial/test",
            qos=1,
            train_path=self.train_path,
            limit=2,
        )

        count = publish_replay(config, client_factory=lambda: client)

        self.assertEqual(count, 2)
        self.assertEqual(client.connected_to, ("broker.local", 1884))
        self.assertTrue(client.disconnected)
        self.assertEqual(len(client.published), 2)
        topic, payload, qos = client.published[0]
        self.assertEqual(topic, "industrial/test")
        self.assertEqual(qos, 1)
        self.assertEqual(json.loads(payload)["cycle"], 1)

    def test_serialize_sensor_message_validates_before_json(self) -> None:
        config = MqttPublisherConfig(train_path=self.train_path, limit=1, dry_run=True)
        output = io.StringIO()
        publish_replay(config, output=output)
        payload = output.getvalue().split("payload=", maxsplit=1)[1].strip()

        self.assertEqual(payload, serialize_sensor_message(next(iter_mock_messages(self.train_path))))

    def test_connect_failure_returns_clear_error_and_disconnects(self) -> None:
        client = MockMqttClient(connect_rc=5)
        config = MqttPublisherConfig(train_path=self.train_path, limit=1)

        with self.assertRaisesRegex(MqttPublisherError, "Failed to connect"):
            publish_replay(config, client_factory=lambda: client)

        self.assertTrue(client.disconnected)

    def test_publish_failure_returns_clear_error_and_disconnects(self) -> None:
        client = MockMqttClient(publish_rc=4)
        config = MqttPublisherConfig(train_path=self.train_path, limit=1)

        with self.assertRaisesRegex(MqttPublisherError, "Failed to publish"):
            publish_replay(config, client_factory=lambda: client)

        self.assertTrue(client.disconnected)

    def test_missing_paho_dependency_is_reported(self) -> None:
        config = MqttPublisherConfig(train_path=self.train_path, limit=0)

        with patch("iot.mqtt_publisher.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(MqttPublisherError, "Missing Paho MQTT dependency"):
                publish_replay(config)

    def test_main_returns_nonzero_for_failure(self) -> None:
        with patch("iot.mqtt_publisher.publish_replay", side_effect=MqttPublisherError("boom")):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = main(["--dry-run", "--limit", "1", "--train-path", str(self.train_path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("MQTT publisher error: boom", stderr.getvalue())


def iter_mock_messages(train_path: Path):
    from iot.local_replay import iter_fd001_training_messages

    return iter_fd001_training_messages(train_path, limit=1)


if __name__ == "__main__":
    unittest.main()
