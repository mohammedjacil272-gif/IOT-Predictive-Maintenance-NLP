"""Ingest validated MQTT sensor messages into InfluxDB.


This module is intentionally separate from local replay and MQTT publishing.

Third-party MQTT and InfluxDB imports are lazy so schema validation and the

existing local tests do not require those runtime dependencies.

"""


from __future__ import annotations


import argparse

from dataclasses import dataclass

import importlib

import json

import os

import sys

from typing import Callable, Mapping, Protocol, Sequence


from .mqtt_publisher import (

    DEFAULT_MQTT_HOST,

    DEFAULT_MQTT_PORT,

    DEFAULT_MQTT_QOS,

    DEFAULT_MQTT_TOPIC,

)

from .sensor_message import (

    FD001_SENSOR_FEATURES,

    SensorMessage,

    SensorMessageValidationError,

    validate_sensor_message,

)


DEFAULT_INFLUX_MEASUREMENT = "fd001_sensor_reading"

SYNTHETIC_EPOCH_NS = 946_684_800_000_000_000

SYNTHETIC_CYCLE_INTERVAL_NS = 1_000_000_000

SYNTHETIC_QUERY_RANGE_START = "2000-01-01T00:00:00Z"


ENV_INFLUX_URL = "IOT_INFLUXDB_URL"

ENV_INFLUX_ORG = "IOT_INFLUXDB_ORG"

ENV_INFLUX_BUCKET = "IOT_INFLUXDB_BUCKET"

ENV_INFLUX_TOKEN = "IOT_INFLUXDB_TOKEN"

ENV_INFLUX_MEASUREMENT = "IOT_INFLUXDB_MEASUREMENT"

ENV_MQTT_HOST = "IOT_MQTT_HOST"

ENV_MQTT_PORT = "IOT_MQTT_PORT"

ENV_MQTT_TOPIC = "IOT_MQTT_TOPIC"

ENV_MQTT_QOS = "IOT_MQTT_QOS"


class InfluxIngestionError(RuntimeError):

    """Raised when MQTT to InfluxDB ingestion cannot proceed."""


class InvalidSensorPayloadError(InfluxIngestionError):

    """Raised when an MQTT payload is not an approved sensor message."""


class InfluxWriteFailure(InfluxIngestionError):

    """Raised when a validated sensor message cannot be persisted."""


class InfluxIngestionRunError(InfluxIngestionError):

    """Raised after a run that observed one or more persistence failures."""


    def __init__(self, message: str, stats: "InfluxIngestionStats") -> None:

        super().__init__(message)

        self.stats = stats


class MqttSubscriberClient(Protocol):

    def connect(self, host: str, port: int) -> int:

        """Connect to an MQTT broker."""


    def subscribe(self, topic: str, qos: int = 0):

        """Subscribe to an MQTT topic."""


    def loop_forever(self) -> int:

        """Run the MQTT network loop until stopped."""


    def disconnect(self) -> int:

        """Disconnect from the MQTT broker."""


class InfluxWriteApi(Protocol):

    def write(self, *, bucket: str, org: str, record: object) -> object:

        """Write one record to InfluxDB."""


class InfluxQueryApi(Protocol):

    def query(self, query: str, org: str) -> object:

        """Run a Flux query."""


class InfluxClient(Protocol):

    def write_api(self, *args: object, **kwargs: object) -> InfluxWriteApi:

        """Return an InfluxDB write API."""


    def query_api(self) -> InfluxQueryApi:

        """Return an InfluxDB query API."""


    def close(self) -> None:

        """Close the InfluxDB client."""


@dataclass(frozen=True)

class InfluxIngestionConfig:

    influx_url: str

    influx_org: str

    influx_bucket: str

    influx_token: str

    mqtt_host: str = DEFAULT_MQTT_HOST

    mqtt_port: int = DEFAULT_MQTT_PORT

    mqtt_topic: str = DEFAULT_MQTT_TOPIC

    mqtt_qos: int = DEFAULT_MQTT_QOS

    measurement: str = DEFAULT_INFLUX_MEASUREMENT


@dataclass

class InfluxIngestionStats:

    messages_received: int = 0

    messages_written: int = 0

    invalid_payloads: int = 0

    write_failures: int = 0

    ingestion_failures: int = 0


def sensor_message_to_point(

    message: SensorMessage,

    *,

    measurement: str = DEFAULT_INFLUX_MEASUREMENT,

    point_factory: Callable[[str], object] | None = None,

) -> object:

    """Convert a validated sensor message into an InfluxDB point.


    The C-MAPSS cycle is an operating-cycle index, not a wall-clock timestamp.

    InfluxDB points use a synthetic timestamp derived only from cycle so writes

    for the same unit/cycle are idempotent and adjacent cycles do not collide.

    """


    normalized = validate_sensor_message(message.to_dict())

    point = (point_factory or _create_point)(measurement)

    point = _call_point_method(point, "tag", "unit_id", str(normalized.unit_id))

    point = _call_point_method(point, "field", "cycle", normalized.cycle)

    for feature in FD001_SENSOR_FEATURES:

        point = _call_point_method(point, "field", feature, normalized.features[feature])

    point = _call_point_method(point, "time", _synthetic_timestamp_ns(normalized))

    return point


def start_ingestion(

    config: InfluxIngestionConfig,

    *,

    mqtt_client_factory: Callable[[], MqttSubscriberClient] | None = None,

    influx_client_factory: Callable[[InfluxIngestionConfig], InfluxClient] | None = None,

    point_factory: Callable[[str], object] | None = None,

    error_handler: Callable[[Exception], None] | None = None,

) -> InfluxIngestionStats:

    """Subscribe to MQTT and write valid sensor messages to InfluxDB."""


    _validate_config(config)

    stats = InfluxIngestionStats()

    influx_client: InfluxClient | None = None

    mqtt_client: MqttSubscriberClient | None = None


    try:

        influx_client = (influx_client_factory or _create_influx_client)(config)

        try:

            mqtt_client = (mqtt_client_factory or _create_paho_client)()

        except Exception:

            _close_influx_client(influx_client)

            influx_client = None

            raise


        write_api = _create_write_api(influx_client)


        def on_message(_client: object, _userdata: object, mqtt_message: object) -> None:

            stats.messages_received += 1

            try:

                sensor_message = _decode_sensor_payload(getattr(mqtt_message, "payload"))

            except InvalidSensorPayloadError as exc:

                stats.invalid_payloads += 1

                if error_handler is not None:

                    _report_error(error_handler, exc)

                return


            try:

                point = sensor_message_to_point(

                    sensor_message,

                    measurement=config.measurement,

                    point_factory=point_factory,

                )

                write_api.write(bucket=config.influx_bucket, org=config.influx_org, record=point)

            except Exception as exc:  # noqa: BLE001 - callback must not kill MQTT loop.

                stats.write_failures += 1

                stats.ingestion_failures += 1

                failure = InfluxWriteFailure("Failed to persist validated sensor message to InfluxDB.")

                failure.__cause__ = exc

                if error_handler is not None:

                    _report_error(error_handler, failure)

            else:

                stats.messages_written += 1


        setattr(mqtt_client, "on_message", on_message)

        _connect_mqtt_client(mqtt_client, config)

        _subscribe_mqtt_client(mqtt_client, config)

        _loop_forever(mqtt_client)

    finally:

        _disconnect_mqtt_client(mqtt_client)

        _close_influx_client(influx_client)


    if stats.ingestion_failures:

        raise InfluxIngestionRunError(

            "InfluxDB ingestion failed to persist "

            f"{stats.ingestion_failures} validated message(s); "

            f"wrote {stats.messages_written}, rejected {stats.invalid_payloads} invalid payload(s).",

            stats,

        )


    return stats


def query_sensor_records(

    config: InfluxIngestionConfig,

    *,

    unit_id: int | None = None,

    limit: int = 10,

    influx_client_factory: Callable[[InfluxIngestionConfig], InfluxClient] | None = None,

) -> list[dict[str, object]]:

    """Query recent stored sensor records back from InfluxDB."""


    _validate_config(config)

    if isinstance(limit, bool) or limit <= 0:

        raise InfluxIngestionError("Query limit must be a positive integer.")

    if unit_id is not None and (isinstance(unit_id, bool) or not isinstance(unit_id, int) or unit_id <= 0):

        raise InfluxIngestionError("Query unit_id must be a positive integer.")


    query = _build_query(config, unit_id=unit_id, limit=limit)

    influx_client = (influx_client_factory or _create_influx_client)(config)

    try:

        tables = influx_client.query_api().query(query, org=config.influx_org)

        return _records_from_query_tables(tables)

    finally:

        _close_influx_client(influx_client)


def parse_config(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> InfluxIngestionConfig:

    """Build ingestion config from CLI arguments and environment variables."""


    source_env = os.environ if env is None else env

    args = _parse_args(argv, source_env)

    config = InfluxIngestionConfig(

        influx_url=args.influx_url,

        influx_org=args.influx_org,

        influx_bucket=args.influx_bucket,

        influx_token=args.influx_token,

        mqtt_host=args.mqtt_host,

        mqtt_port=args.mqtt_port,

        mqtt_topic=args.mqtt_topic,

        mqtt_qos=args.mqtt_qos,

        measurement=args.measurement,

    )

    _validate_config(config)

    return config


def main(argv: Sequence[str] | None = None) -> int:

    try:

        args = _parse_args(argv, os.environ)

        config = parse_config(argv)

        if args.query:

            rows = query_sensor_records(config, unit_id=args.query_unit_id, limit=args.query_limit)

            print(json.dumps(rows, separators=(",", ":"), sort_keys=True, default=str))

        else:

            stats = start_ingestion(config)

            print(f"InfluxDB sensor ingestion stopped after writing {stats.messages_written} messages.")

    except (InfluxIngestionError, OSError, ValueError) as exc:

        print(f"InfluxDB ingestion error: {exc}", file=sys.stderr)

        return 1

    return 0


def _decode_sensor_payload(payload: object) -> SensorMessage:

    try:

        if isinstance(payload, bytes):

            text = payload.decode("utf-8")

        elif isinstance(payload, str):

            text = payload

        else:

            raise InvalidSensorPayloadError("MQTT payload must be UTF-8 bytes or text.")

        decoded = json.loads(text)

        return validate_sensor_message(decoded)

    except UnicodeDecodeError as exc:

        raise InvalidSensorPayloadError("MQTT payload must be UTF-8 bytes or text.") from exc

    except json.JSONDecodeError as exc:

        raise InvalidSensorPayloadError("MQTT payload must be valid JSON.") from exc

    except SensorMessageValidationError as exc:

        raise InvalidSensorPayloadError(f"MQTT payload is not an approved sensor message: {exc}") from exc


def _parse_args(argv: Sequence[str] | None, source_env: Mapping[str, str]) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Ingest validated MQTT sensor messages into InfluxDB.")

    parser.add_argument("--influx-url", default=source_env.get(ENV_INFLUX_URL, ""))

    parser.add_argument("--influx-org", default=source_env.get(ENV_INFLUX_ORG, ""))

    parser.add_argument("--influx-bucket", default=source_env.get(ENV_INFLUX_BUCKET, ""))

    parser.add_argument("--influx-token", default=source_env.get(ENV_INFLUX_TOKEN, ""))

    parser.add_argument("--measurement", default=source_env.get(ENV_INFLUX_MEASUREMENT, DEFAULT_INFLUX_MEASUREMENT))

    parser.add_argument("--mqtt-host", default=source_env.get(ENV_MQTT_HOST, DEFAULT_MQTT_HOST))

    parser.add_argument("--mqtt-port", type=int, default=_env_int(source_env, ENV_MQTT_PORT, DEFAULT_MQTT_PORT))

    parser.add_argument("--mqtt-topic", default=source_env.get(ENV_MQTT_TOPIC, DEFAULT_MQTT_TOPIC))

    parser.add_argument("--mqtt-qos", type=int, default=_env_int(source_env, ENV_MQTT_QOS, DEFAULT_MQTT_QOS))

    parser.add_argument("--query", action="store_true", help="Query recent records instead of starting ingestion.")

    parser.add_argument("--query-unit-id", type=int, default=None, help="Optional unit_id tag filter for --query.")

    parser.add_argument("--query-limit", type=int, default=10, help="Maximum records to return for --query.")

    return parser.parse_args(argv)


def _create_paho_client() -> MqttSubscriberClient:

    try:

        mqtt = importlib.import_module("paho.mqtt.client")

    except ImportError as exc:

        raise InfluxIngestionError(

            "Missing Paho MQTT dependency. Install it with: python -m pip install paho-mqtt"

        ) from exc


    callback_api_version = getattr(getattr(mqtt, "CallbackAPIVersion", object()), "VERSION2", None)

    if callback_api_version is None:

        return mqtt.Client()

    return mqtt.Client(callback_api_version=callback_api_version)


def _create_influx_client(config: InfluxIngestionConfig) -> InfluxClient:

    try:

        influxdb_client = importlib.import_module("influxdb_client")

    except ImportError as exc:

        raise InfluxIngestionError(

            "Missing InfluxDB dependency. Install it with: python -m pip install influxdb-client"

        ) from exc


    try:

        return influxdb_client.InfluxDBClient(

            url=config.influx_url,

            token=config.influx_token,

            org=config.influx_org,

        )

    except Exception as exc:  # noqa: BLE001 - avoid leaking token details from client internals.

        safe_url = _redact_secret(config.influx_url, config.influx_token)

        raise InfluxIngestionError(

            "Failed to create InfluxDB client "

            f"for {safe_url}; check URL, organization, token presence, and network/TLS settings "

            f"(client error type: {type(exc).__name__})."

        ) from exc


def _create_point(measurement: str) -> object:

    try:

        influxdb_client = importlib.import_module("influxdb_client")

    except ImportError as exc:

        raise InfluxIngestionError(

            "Missing InfluxDB dependency. Install it with: python -m pip install influxdb-client"

        ) from exc

    return influxdb_client.Point(measurement)


def _create_write_api(client: InfluxClient) -> InfluxWriteApi:

    try:

        write_api_module = importlib.import_module("influxdb_client.client.write_api")

    except ImportError:

        return client.write_api()

    return client.write_api(write_options=write_api_module.SYNCHRONOUS)


def _connect_mqtt_client(client: MqttSubscriberClient, config: InfluxIngestionConfig) -> None:

    try:

        result = client.connect(config.mqtt_host, config.mqtt_port)

    except Exception as exc:  # noqa: BLE001 - surface third-party client failures clearly.

        raise InfluxIngestionError(

            f"Failed to connect to MQTT broker {config.mqtt_host}:{config.mqtt_port}: {exc}"

        ) from exc


    rc = _result_code(result)

    if rc != 0:

        raise InfluxIngestionError(

            f"Failed to connect to MQTT broker {config.mqtt_host}:{config.mqtt_port}; return code {rc}."

        )


def _subscribe_mqtt_client(client: MqttSubscriberClient, config: InfluxIngestionConfig) -> None:

    try:

        result = client.subscribe(config.mqtt_topic, qos=config.mqtt_qos)

    except Exception as exc:  # noqa: BLE001 - surface third-party client failures clearly.

        raise InfluxIngestionError(f"Failed to subscribe to MQTT topic {config.mqtt_topic}: {exc}") from exc


    rc = _result_code(result)

    if rc != 0:

        raise InfluxIngestionError(f"Failed to subscribe to MQTT topic {config.mqtt_topic}; return code {rc}.")


def _loop_forever(client: MqttSubscriberClient) -> None:

    try:

        result = client.loop_forever()

    except KeyboardInterrupt:

        return

    except Exception as exc:  # noqa: BLE001 - surface third-party client failures clearly.

        raise InfluxIngestionError(f"MQTT loop failed: {exc}") from exc


    rc = _result_code(result)

    if rc != 0:

        raise InfluxIngestionError(f"MQTT loop stopped with return code {rc}.")


def _disconnect_mqtt_client(client: object) -> None:

    if client is None:

        return

    disconnect = getattr(client, "disconnect", None)

    if callable(disconnect):

        try:

            disconnect()

        except Exception:

            return


def _close_influx_client(client: object) -> None:

    if client is None:

        return

    close = getattr(client, "close", None)

    if callable(close):

        close()


def _result_code(result: object) -> int:

    if isinstance(result, int):

        return result

    if isinstance(result, tuple) and result and isinstance(result[0], int):

        return result[0]

    rc = getattr(result, "rc", None)

    if isinstance(rc, int):

        return rc

    return 0


def _call_point_method(point: object, method_name: str, *args: object) -> object:

    method = getattr(point, method_name)

    result = method(*args)

    return point if result is None else result


def _synthetic_timestamp_ns(message: SensorMessage) -> int:

    return SYNTHETIC_EPOCH_NS + ((message.cycle - 1) * SYNTHETIC_CYCLE_INTERVAL_NS)


def _build_query(config: InfluxIngestionConfig, *, unit_id: int | None, limit: int) -> str:

    query = [

        f'from(bucket: "{_escape_flux_string(config.influx_bucket)}")',

        f"  |> range(start: {SYNTHETIC_QUERY_RANGE_START})",

        f'  |> filter(fn: (r) => r._measurement == "{_escape_flux_string(config.measurement)}")',

    ]

    if unit_id is not None:

        query.append(f'  |> filter(fn: (r) => r.unit_id == "{unit_id}")')

    query.extend(

        [

            '  |> filter(fn: (r) => r._field == "cycle" or '

            + " or ".join(f'r._field == "{feature}"' for feature in FD001_SENSOR_FEATURES)

            + ")",

            '  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")',

            '  |> sort(columns: ["_time"], desc: true)',

            f"  |> limit(n: {limit})",

        ]

    )

    return "\n".join(query)


def _escape_flux_string(value: str) -> str:

    return value.replace("\\", "\\\\").replace('"', '\\\\"')


def _report_error(error_handler: Callable[[Exception], None], error: Exception) -> None:

    try:

        error_handler(error)

    except Exception:

        return


def _records_from_query_tables(tables: object) -> list[dict[str, object]]:

    records: list[dict[str, object]] = []

    for table in tables:

        for record in getattr(table, "records", []):

            values = getattr(record, "values", None)

            if isinstance(values, Mapping):

                records.append(dict(values))

            elif hasattr(record, "values") and callable(record.values):

                records.append(dict(record.values()))

    return records


def _validate_config(config: InfluxIngestionConfig) -> None:

    for name in ("influx_url", "influx_org", "influx_bucket", "influx_token", "measurement", "mqtt_host", "mqtt_topic"):

        if not isinstance(getattr(config, name), str):

            raise InfluxIngestionError(f"InfluxDB/MQTT configuration field {name} must be text.")

    if not config.influx_url:

        raise InfluxIngestionError(f"{ENV_INFLUX_URL} or --influx-url is required.")

    if not config.influx_org:

        raise InfluxIngestionError(f"{ENV_INFLUX_ORG} or --influx-org is required.")

    if not config.influx_bucket:

        raise InfluxIngestionError(f"{ENV_INFLUX_BUCKET} or --influx-bucket is required.")

    if not config.influx_token:

        raise InfluxIngestionError(f"{ENV_INFLUX_TOKEN} or --influx-token is required.")

    if not config.measurement:

        raise InfluxIngestionError("InfluxDB measurement must not be empty.")

    if not config.mqtt_host:

        raise InfluxIngestionError("MQTT broker host must not be empty.")

    if isinstance(config.mqtt_port, bool) or not isinstance(config.mqtt_port, int) or not 1 <= config.mqtt_port <= 65535:

        raise InfluxIngestionError("MQTT broker port must be between 1 and 65535.")

    if not config.mqtt_topic:

        raise InfluxIngestionError("MQTT topic must not be empty.")

    if isinstance(config.mqtt_qos, bool) or not isinstance(config.mqtt_qos, int) or config.mqtt_qos not in (0, 1, 2):

        raise InfluxIngestionError("MQTT QoS must be 0, 1, or 2.")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:

    raw_value = env.get(name)

    if raw_value is None:

        return default

    try:

        return int(raw_value)

    except ValueError as exc:

        raise InfluxIngestionError(f"{name} must be an integer.") from exc


def _redact_secret(value: str, secret: str) -> str:

    if not secret:

        return value

    return value.replace(secret, "<redacted>")


if __name__ == "__main__":

    raise SystemExit(main())
