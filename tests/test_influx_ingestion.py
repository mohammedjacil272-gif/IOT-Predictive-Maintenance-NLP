import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from iot.influx_ingestion import (
    DEFAULT_INFLUX_MEASUREMENT,
    InfluxIngestionConfig,
    InfluxIngestionError,
    InfluxIngestionRunError,
    InfluxIngestionStats,
    InfluxWriteFailure,
    SYNTHETIC_CYCLE_INTERVAL_NS,
    SYNTHETIC_EPOCH_NS,
    main,
    parse_config,
    query_sensor_records,
    sensor_message_to_point,
    start_ingestion,
)
from iot.influx_ingestion import _build_query
from iot.sensor_message import FD001_SENSOR_FEATURES, validate_sensor_message


def valid_payload(*, cycle: int = 1) -> dict[str, int | float]:
    payload: dict[str, int | float] = {"unit_id": 1, "cycle": cycle}
    for index, feature in enumerate(FD001_SENSOR_FEATURES, start=1):
        payload[feature] = float(index)
    return payload


class FakePoint:
    def __init__(self, measurement: str) -> None:
        self.measurement = measurement
        self.tags: dict[str, str] = {}
        self.fields: dict[str, int | float] = {}
        self.timestamp_ns: int | None = None

    def tag(self, key: str, value: str) -> "FakePoint":
        self.tags[key] = value
        return self

    def field(self, key: str, value: int | float) -> "FakePoint":
        self.fields[key] = value
        return self

    def time(self, timestamp_ns: int) -> "FakePoint":
        self.timestamp_ns = timestamp_ns
        return self


class FakeMqttMessage:
    def __init__(self, payload: str | bytes) -> None:
        self.payload = payload


class FakeMqttClient:
    def __init__(
        self,
        payloads: list[str | bytes] | None = None,
        *,
        connect_rc: int = 0,
        subscribe_rc: int = 0,
        loop_rc: int = 0,
        disconnect_raises: bool = False,
    ) -> None:
        self.payloads = payloads or []
        self.connect_rc = connect_rc
        self.subscribe_rc = subscribe_rc
        self.loop_rc = loop_rc
        self.disconnect_raises = disconnect_raises
        self.connected_to: tuple[str, int] | None = None
        self.subscribed_to: tuple[str, int] | None = None
        self.disconnected = False
        self.on_message = None

    def connect(self, host: str, port: int) -> int:
        self.connected_to = (host, port)
        return self.connect_rc

    def subscribe(self, topic: str, qos: int = 0) -> tuple[int, int]:
        self.subscribed_to = (topic, qos)
        return (self.subscribe_rc, 1)

    def loop_forever(self) -> int:
        for payload in self.payloads:
            if callable(self.on_message):
                self.on_message(self, None, FakeMqttMessage(payload))
        return self.loop_rc

    def disconnect(self) -> int:
        self.disconnected = True
        if self.disconnect_raises:
            raise RuntimeError("disconnect failed")
        return 0


class FakeWriteApi:
    def __init__(self, store: list[FakePoint], *, failures: list[Exception] | None = None) -> None:
        self.store = store
        self.failures = failures or []
        self.calls: list[tuple[str, str, FakePoint]] = []

    def write(self, *, bucket: str, org: str, record: FakePoint) -> None:
        if self.failures:
            raise self.failures.pop(0)
        self.calls.append((bucket, org, record))
        self.store.append(record)


class FakeQueryApi:
    def __init__(self, store: list[FakePoint]) -> None:
        self.store = store
        self.queries: list[tuple[str, str]] = []

    def query(self, query: str, org: str) -> list[SimpleNamespace]:
        self.queries.append((query, org))
        records = []
        for point in self.store:
            values = {
                "_measurement": point.measurement,
                "_time": point.timestamp_ns,
                "unit_id": point.tags["unit_id"],
                **point.fields,
            }
            records.append(SimpleNamespace(values=values))
        return [SimpleNamespace(records=records)]



class FakeInfluxClient:
    def __init__(
        self,
        store: list[FakePoint] | None = None,
        *,
        write_failures: list[Exception] | None = None,
    ) -> None:
        self.store = store if store is not None else []
        self.write_api_instance = FakeWriteApi(
            self.store,
            failures=write_failures,
        )
        self.query_api_instance = FakeQueryApi(self.store)
        self.closed = False

    def write_api(self, *args: object, **kwargs: object) -> FakeWriteApi:
        """Return the fake write API, accepting production client options."""
        return self.write_api_instance

    def query_api(self) -> FakeQueryApi:
        return self.query_api_instance

    def close(self) -> None:
        self.closed = True



def test_config(**overrides: object) -> InfluxIngestionConfig:
    values: dict[str, object] = {
        "influx_url": "http://localhost:8086",
        "influx_org": "local-org",
        "influx_bucket": "iot-sensors",
        "influx_token": "secret-token",
    }
    values.update(overrides)
    return InfluxIngestionConfig(**values)


class TestInfluxIngestion(unittest.TestCase):
    def test_parse_config_uses_env_and_cli_overrides(self) -> None:
        env = {
            "IOT_INFLUXDB_URL": "http://influx:8086",
            "IOT_INFLUXDB_ORG": "factory",
            "IOT_INFLUXDB_BUCKET": "sensors",
            "IOT_INFLUXDB_TOKEN": "env-token",
            "IOT_MQTT_HOST": "broker.local",
            "IOT_MQTT_PORT": "1884",
            "IOT_MQTT_TOPIC": "factory/line-a",
            "IOT_MQTT_QOS": "1",
        }

        config = parse_config(["--mqtt-topic", "industrial/sensors"], env=env)

        self.assertEqual(config.influx_url, "http://influx:8086")
        self.assertEqual(config.influx_org, "factory")
        self.assertEqual(config.influx_bucket, "sensors")
        self.assertEqual(config.influx_token, "env-token")
        self.assertEqual(config.mqtt_host, "broker.local")
        self.assertEqual(config.mqtt_port, 1884)
        self.assertEqual(config.mqtt_topic, "industrial/sensors")
        self.assertEqual(config.mqtt_qos, 1)

    def test_config_requires_influx_token_without_printing_secret(self) -> None:
        with self.assertRaisesRegex(InfluxIngestionError, "IOT_INFLUXDB_TOKEN") as error:
            parse_config(
                [
                    "--influx-url",
                    "http://localhost:8086",
                    "--influx-org",
                    "local-org",
                    "--influx-bucket",
                    "iot-sensors",
                ],
                env={},
            )

        self.assertNotIn("secret-token", str(error.exception))

    def test_invalid_mqtt_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(InfluxIngestionError, "port"):
            parse_config(
                [
                    "--influx-url",
                    "http://localhost:8086",
                    "--influx-org",
                    "local-org",
                    "--influx-bucket",
                    "iot-sensors",
                    "--influx-token",
                    "secret-token",
                    "--mqtt-port",
                    "70000",
                ],
                env={},
            )

    def test_direct_config_rejects_non_integer_mqtt_settings(self) -> None:
        with self.assertRaisesRegex(InfluxIngestionError, "port"):
            start_ingestion(test_config(mqtt_port="1883"))

        with self.assertRaisesRegex(InfluxIngestionError, "QoS"):
            start_ingestion(test_config(mqtt_qos=True))

    def test_flux_string_escaping_handles_backslashes_and_quotes(self) -> None:
        query = _build_query(
            test_config(influx_bucket='bucket\\name"x', measurement='measurement\\name"x'),
            unit_id=None,
            limit=1,
        )

        self.assertIn('from(bucket: "bucket' + "\\\\" + 'name' + "\\\\" + '"x")', query)
        self.assertIn('r._measurement == "measurement' + "\\\\" + 'name' + "\\\\" + '"x"', query)

    def test_sensor_message_to_point_preserves_schema_with_deterministic_timestamp(self) -> None:
        message = validate_sensor_message(valid_payload(cycle=30))

        point = sensor_message_to_point(message, point_factory=FakePoint)

        self.assertEqual(point.measurement, DEFAULT_INFLUX_MEASUREMENT)
        self.assertEqual(point.tags, {"unit_id": "1"})
        self.assertEqual(point.fields["cycle"], 30)
        self.assertEqual([key for key in point.fields if key != "cycle"], list(FD001_SENSOR_FEATURES))
        self.assertNotIn("timestamp", point.fields)
        self.assertNotIn("rul", point.fields)
        self.assertEqual(point.timestamp_ns, SYNTHETIC_EPOCH_NS + (29 * SYNTHETIC_CYCLE_INTERVAL_NS))

    def test_repeated_unit_cycle_uses_same_point_identity(self) -> None:
        message = validate_sensor_message(valid_payload(cycle=7))

        first = sensor_message_to_point(message, point_factory=FakePoint)
        second = sensor_message_to_point(message, point_factory=FakePoint)

        self.assertEqual(first.tags, second.tags)
        self.assertEqual(first.timestamp_ns, second.timestamp_ns)
        self.assertEqual(first.fields["cycle"], second.fields["cycle"])

    def test_distinct_cycles_for_same_unit_do_not_collide(self) -> None:
        first = sensor_message_to_point(validate_sensor_message(valid_payload(cycle=7)), point_factory=FakePoint)
        second = sensor_message_to_point(validate_sensor_message(valid_payload(cycle=8)), point_factory=FakePoint)

        self.assertEqual(first.tags, second.tags)
        self.assertNotEqual(first.timestamp_ns, second.timestamp_ns)

    def test_valid_mqtt_payload_is_written_and_can_be_queried_back(self) -> None:
        store: list[FakePoint] = []
        mqtt_client = FakeMqttClient([json.dumps(valid_payload(cycle=2))])
        influx_client = FakeInfluxClient(store)
        config = test_config()

        stats = start_ingestion(
            config,
            mqtt_client_factory=lambda: mqtt_client,
            influx_client_factory=lambda _config: influx_client,
            point_factory=FakePoint,
        )
        records = query_sensor_records(config, influx_client_factory=lambda _config: FakeInfluxClient(store))

        self.assertEqual(stats.messages_received, 1)
        self.assertEqual(stats.messages_written, 1)
        self.assertEqual(mqtt_client.connected_to, ("localhost", 1883))
        self.assertEqual(mqtt_client.subscribed_to, ("industrial/sensors", 0))
        self.assertTrue(mqtt_client.disconnected)
        self.assertTrue(influx_client.closed)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["unit_id"], "1")
        self.assertEqual(records[0]["cycle"], 2)
        self.assertEqual(records[0]["sensor_21"], 17.0)

    def test_invalid_payload_is_counted_and_not_written(self) -> None:
        payload = valid_payload()
        payload["timestamp"] = 123.0
        mqtt_client = FakeMqttClient([json.dumps(payload)])
        influx_client = FakeInfluxClient([])
        errors: list[Exception] = []

        stats = start_ingestion(
            test_config(),
            mqtt_client_factory=lambda: mqtt_client,
            influx_client_factory=lambda _config: influx_client,
            point_factory=FakePoint,
            error_handler=errors.append,
        )

        self.assertEqual(stats.messages_received, 1)
        self.assertEqual(stats.invalid_payloads, 1)
        self.assertEqual(stats.messages_written, 0)
        self.assertEqual(influx_client.write_api_instance.calls, [])
        self.assertTrue(errors)

    def test_error_handler_failure_does_not_stop_mqtt_loop(self) -> None:
        payloads = [json.dumps({"bad": True}), json.dumps(valid_payload(cycle=2))]
        mqtt_client = FakeMqttClient(payloads)
        influx_client = FakeInfluxClient([])

        def failing_error_handler(_error: Exception) -> None:
            raise RuntimeError("error reporting failed")

        stats = start_ingestion(
            test_config(),
            mqtt_client_factory=lambda: mqtt_client,
            influx_client_factory=lambda _config: influx_client,
            point_factory=FakePoint,
            error_handler=failing_error_handler,
        )

        self.assertEqual(stats.invalid_payloads, 1)
        self.assertEqual(stats.messages_written, 1)

    def test_mqtt_connection_failure_is_reported_and_clients_close(self) -> None:
        mqtt_client = FakeMqttClient(connect_rc=5)
        influx_client = FakeInfluxClient([])

        with self.assertRaisesRegex(InfluxIngestionError, "Failed to connect"):
            start_ingestion(
                test_config(),
                mqtt_client_factory=lambda: mqtt_client,
                influx_client_factory=lambda _config: influx_client,
            )

        self.assertTrue(mqtt_client.disconnected)
        self.assertTrue(influx_client.closed)

    def test_write_failure_is_counted_and_reported_after_loop(self) -> None:
        mqtt_client = FakeMqttClient([json.dumps(valid_payload(cycle=1)), json.dumps(valid_payload(cycle=2))])
        influx_client = FakeInfluxClient([], write_failures=[RuntimeError("write failed")])
        errors: list[Exception] = []

        with self.assertRaisesRegex(InfluxIngestionRunError, "failed to persist") as error:
            start_ingestion(
                test_config(),
                mqtt_client_factory=lambda: mqtt_client,
                influx_client_factory=lambda _config: influx_client,
                point_factory=FakePoint,
                error_handler=errors.append,
            )

        stats = error.exception.stats
        self.assertEqual(stats.messages_received, 2)
        self.assertEqual(stats.write_failures, 1)
        self.assertEqual(stats.ingestion_failures, 1)
        self.assertEqual(stats.messages_written, 1)
        self.assertIsInstance(errors[0], InfluxWriteFailure)
        self.assertEqual(len(influx_client.write_api_instance.calls), 1)

    def test_write_value_error_is_not_classified_as_invalid_payload(self) -> None:
        mqtt_client = FakeMqttClient([json.dumps(valid_payload())])
        influx_client = FakeInfluxClient([], write_failures=[ValueError("bad write")])
        errors: list[Exception] = []

        with self.assertRaises(InfluxIngestionRunError) as error:
            start_ingestion(
                test_config(),
                mqtt_client_factory=lambda: mqtt_client,
                influx_client_factory=lambda _config: influx_client,
                point_factory=FakePoint,
                error_handler=errors.append,
            )

        stats = error.exception.stats
        self.assertEqual(stats.messages_received, 1)
        self.assertEqual(stats.write_failures, 1)
        self.assertEqual(stats.ingestion_failures, 1)
        self.assertEqual(stats.invalid_payloads, 0)
        self.assertEqual(stats.messages_written, 0)
        self.assertIsInstance(errors[0], InfluxWriteFailure)

    def test_write_type_error_is_not_classified_as_invalid_payload(self) -> None:
        mqtt_client = FakeMqttClient([json.dumps(valid_payload())])
        influx_client = FakeInfluxClient([], write_failures=[TypeError("bad write")])

        with self.assertRaises(InfluxIngestionRunError) as error:
            start_ingestion(
                test_config(),
                mqtt_client_factory=lambda: mqtt_client,
                influx_client_factory=lambda _config: influx_client,
                point_factory=FakePoint,
            )

        stats = error.exception.stats
        self.assertEqual(stats.write_failures, 1)
        self.assertEqual(stats.ingestion_failures, 1)
        self.assertEqual(stats.invalid_payloads, 0)

    def test_influx_client_is_closed_if_mqtt_client_factory_fails(self) -> None:
        influx_client = FakeInfluxClient([])

        with self.assertRaisesRegex(RuntimeError, "mqtt factory failed"):
            start_ingestion(
                test_config(),
                mqtt_client_factory=lambda: (_ for _ in ()).throw(RuntimeError("mqtt factory failed")),
                influx_client_factory=lambda _config: influx_client,
            )

        self.assertTrue(influx_client.closed)

    def test_disconnect_exception_does_not_prevent_influx_cleanup(self) -> None:
        mqtt_client = FakeMqttClient([], disconnect_raises=True)
        influx_client = FakeInfluxClient([])

        stats = start_ingestion(
            test_config(),
            mqtt_client_factory=lambda: mqtt_client,
            influx_client_factory=lambda _config: influx_client,
        )

        self.assertEqual(stats.messages_received, 0)
        self.assertTrue(mqtt_client.disconnected)
        self.assertTrue(influx_client.closed)

    def test_client_creation_failure_does_not_print_token(self) -> None:
        token = "fake-token-secret"
        config = test_config(influx_token=token)

        fake_module = SimpleNamespace(
            InfluxDBClient=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError(f"cannot connect with token {token}")
            )
        )

        def fake_import(name: str) -> object:
            if name == "influxdb_client":
                return fake_module
            raise ImportError(name)

        with patch("iot.influx_ingestion.importlib.import_module", side_effect=fake_import):
            with self.assertRaisesRegex(InfluxIngestionError, "Failed to create InfluxDB client") as error:
                start_ingestion(config, mqtt_client_factory=lambda: FakeMqttClient([]))

        self.assertNotIn(token, str(error.exception))

    def test_main_client_creation_failure_does_not_print_token(self) -> None:
        token = "fake-token-secret"

        def fail_create(_config: InfluxIngestionConfig) -> FakeInfluxClient:
            raise InfluxIngestionError(
                "Failed to create InfluxDB client; check token presence and network/TLS settings."
            )

        stderr = io.StringIO()
        stdout = io.StringIO()
        with patch("iot.influx_ingestion._create_influx_client", side_effect=fail_create):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = main(
                    [
                        "--influx-url",
                        "http://localhost:8086",
                        "--influx-org",
                        "local-org",
                        "--influx-bucket",
                        "iot-sensors",
                        "--influx-token",
                        token,
                    ]
                )

        self.assertEqual(result, 1)
        self.assertNotIn(token, stdout.getvalue())
        self.assertNotIn(token, stderr.getvalue())
        self.assertIn("InfluxDB ingestion error", stderr.getvalue())

    def test_main_returns_nonzero_for_persistence_failures(self) -> None:
        stats = InfluxIngestionStats(messages_received=1, write_failures=1, ingestion_failures=1)
        failure = InfluxIngestionRunError("InfluxDB ingestion failed to persist 1 validated message(s).", stats)
        stderr = io.StringIO()
        stdout = io.StringIO()

        with patch("iot.influx_ingestion.start_ingestion", side_effect=failure):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = main(
                    [
                        "--influx-url",
                        "http://localhost:8086",
                        "--influx-org",
                        "local-org",
                        "--influx-bucket",
                        "iot-sensors",
                        "--influx-token",
                        "secret-token",
                    ]
                )

        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("failed to persist", stderr.getvalue())

    def test_main_query_serializes_non_json_values(self) -> None:
        stdout = io.StringIO()
        with patch("iot.influx_ingestion.query_sensor_records", return_value=[{"_time": object()}]):
            with redirect_stdout(stdout):
                result = main(
                    [
                        "--influx-url", "http://localhost:8086",
                        "--influx-org", "IOT",
                        "--influx-bucket", "industrial_sensors",
                        "--influx-token", "secret-token",
                        "--query",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertIn('"_time":"<object object at', stdout.getvalue())

    def test_missing_influx_dependency_is_lazy_and_reported(self) -> None:
        config = test_config()

        with patch("iot.influx_ingestion.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(InfluxIngestionError, "Missing InfluxDB dependency"):
                sensor_message_to_point(validate_sensor_message(valid_payload()))

        with patch("iot.influx_ingestion.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(InfluxIngestionError, "Missing InfluxDB dependency"):
                start_ingestion(config, mqtt_client_factory=lambda: FakeMqttClient([]))


if __name__ == "__main__":
    unittest.main()
