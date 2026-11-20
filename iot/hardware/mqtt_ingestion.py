"""MQTT subscriber and validated hardware telemetry ingestion service."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import os
from threading import Event
from typing import Callable, Mapping, Protocol

from .influx_storage import InfluxStorageConfig, write_telemetry
from .message import HardwareTelemetry, HardwareTelemetryValidationError, deserialize_payload

LOGGER = logging.getLogger(__name__)
DEFAULT_MQTT_HOST = "localhost"
DEFAULT_MQTT_PORT = 1883
DEFAULT_HARDWARE_TOPIC = "maintainiq/hardware/telemetry"
DEFAULT_MQTT_QOS = 1
ENV_MQTT_HOST = "IOT_HARDWARE_MQTT_HOST"
ENV_MQTT_PORT = "IOT_HARDWARE_MQTT_PORT"
ENV_MQTT_TOPIC = "IOT_HARDWARE_MQTT_TOPIC"
ENV_MQTT_QOS = "IOT_HARDWARE_MQTT_QOS"
ENV_MQTT_USERNAME = "IOT_HARDWARE_MQTT_USERNAME"
ENV_MQTT_PASSWORD = "IOT_HARDWARE_MQTT_PASSWORD"


class HardwareMqttError(RuntimeError):
    """Raised when the hardware MQTT subscriber cannot start."""


class MqttClient(Protocol):
    on_message: Callable[..., None]
    def connect(self, host: str, port: int, keepalive: int = 60) -> object: ...
    def subscribe(self, topic: str, qos: int = 0) -> object: ...
    def loop_forever(self, **kwargs: object) -> object: ...
    def disconnect(self) -> object: ...


@dataclass(frozen=True)
class MqttIngestionConfig:
    host: str = DEFAULT_MQTT_HOST
    port: int = DEFAULT_MQTT_PORT
    topic: str = DEFAULT_HARDWARE_TOPIC
    qos: int = DEFAULT_MQTT_QOS
    username: str = ""
    password: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "MqttIngestionConfig":
        values = os.environ if env is None else env
        config = cls(values.get(ENV_MQTT_HOST, DEFAULT_MQTT_HOST),
                     _env_int(values, ENV_MQTT_PORT, DEFAULT_MQTT_PORT),
                     values.get(ENV_MQTT_TOPIC, DEFAULT_HARDWARE_TOPIC),
                     _env_int(values, ENV_MQTT_QOS, DEFAULT_MQTT_QOS),
                     values.get(ENV_MQTT_USERNAME, ""), values.get(ENV_MQTT_PASSWORD, ""))
        config.validate()
        return config

    def validate(self) -> None:
        if not self.host or not 1 <= self.port <= 65535 or not self.topic or self.qos not in (0, 1, 2):
            raise HardwareMqttError("Invalid hardware MQTT configuration.")


@dataclass
class IngestionStats:
    received: int = 0
    accepted: int = 0
    invalid: int = 0
    duplicates: int = 0
    write_errors: int = 0


class SequenceTracker:
    """In-memory deduplication by device/session/sequence; resets with process."""

    def __init__(self) -> None:
        self._highest: dict[tuple[str, str], int] = {}

    def is_duplicate(self, telemetry: HardwareTelemetry) -> bool:
        key = (telemetry.device_id, telemetry.session_id)
        return telemetry.sequence_number <= self._highest.get(key, 0)

    def mark_stored(self, telemetry: HardwareTelemetry) -> None:
        key = (telemetry.device_id, telemetry.session_id)
        self._highest[key] = telemetry.sequence_number


def process_payload(raw: str | bytes, *, tracker: SequenceTracker, stats: IngestionStats,
                    writer: Callable[[HardwareTelemetry], None]) -> bool:
    stats.received += 1
    try:
        telemetry = deserialize_payload(raw)
    except HardwareTelemetryValidationError as exc:
        stats.invalid += 1
        LOGGER.warning("Rejected invalid hardware telemetry: %s", exc)
        return False
    if tracker.is_duplicate(telemetry):
        stats.duplicates += 1
        LOGGER.info("Ignored duplicate or out-of-order hardware sequence for device %s", telemetry.device_id)
        return False
    try:
        writer(telemetry)
    except Exception as exc:  # keep MQTT callback alive; caller can inspect counters/logs
        stats.write_errors += 1
        LOGGER.error("Hardware telemetry write failed: %s", exc)
        return False
    tracker.mark_stored(telemetry)
    stats.accepted += 1
    return True


def start_ingestion(mqtt_config: MqttIngestionConfig, influx_config: InfluxStorageConfig, *,
                    client_factory: Callable[[], MqttClient] | None = None,
                    writer: Callable[[HardwareTelemetry], None] | None = None,
                    stop_event: Event | None = None) -> IngestionStats:
    mqtt_config.validate()
    influx_config.validate()
    factory = client_factory or _create_mqtt_client
    client = factory()
    stats = IngestionStats()
    tracker = SequenceTracker()
    stop = stop_event or Event()
    storage_writer = writer or (lambda message: write_telemetry(message, influx_config))

    def on_message(_client: object, _userdata: object, message: object) -> None:
        process_payload(getattr(message, "payload", b""), tracker=tracker, stats=stats, writer=storage_writer)

    def on_connect(connected_client: object, _userdata: object, *_args: object) -> None:
        # Subscribe after every successful CONNACK so reconnects restore the stream.
        reason = _args[1] if len(_args) > 1 else 0  # v2: flags, reason_code, properties; v1: flags, rc
        failed = getattr(reason, "is_failure", False)
        if failed or (isinstance(reason, int) and reason != 0):
            LOGGER.error("Hardware MQTT broker rejected connection (reason=%s)", reason)
            return
        try:
            connected_client.subscribe(mqtt_config.topic, qos=mqtt_config.qos)
        except Exception as exc:
            LOGGER.error("Hardware MQTT subscription failed: %s", exc)

    client.on_message = on_message
    client.on_connect = on_connect
    try:
        username = mqtt_config.username
        if username:
            setter = getattr(client, "username_pw_set", None)
            if not callable(setter):
                raise HardwareMqttError("MQTT client does not support username/password authentication.")
            setter(username, mqtt_config.password or None)
        client.connect(mqtt_config.host, mqtt_config.port, keepalive=60)
        LOGGER.info("Listening for hardware telemetry on %s", mqtt_config.topic)
        if stop_event is None:
            client.loop_forever()
        else:
            # Paho loop_start reconnects automatically; stop event gives services/tests controlled shutdown.
            client.loop_start()
            stop.wait()
            client.loop_stop()
    except Exception as exc:
        raise HardwareMqttError("Hardware MQTT ingestion stopped after a connection or loop error.") from exc
    finally:
        try:
            client.disconnect()
        except Exception:
            LOGGER.debug("MQTT disconnect failed during cleanup", exc_info=True)
    return stats


def _create_mqtt_client() -> MqttClient:
    try:
        mqtt = importlib.import_module("paho.mqtt.client")
    except ImportError as exc:
        raise HardwareMqttError("Missing MQTT dependency. Install paho-mqtt.") from exc
    version = getattr(getattr(mqtt, "CallbackAPIVersion", object()), "VERSION2", None)
    return mqtt.Client(callback_api_version=version) if version is not None else mqtt.Client()


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, default))
    except (ValueError, TypeError) as exc:
        raise HardwareMqttError(f"{name} must be an integer.") from exc


def main() -> int:
    try:
        start_ingestion(MqttIngestionConfig.from_env(), InfluxStorageConfig.from_env())
    except Exception as exc:
        LOGGER.error("Hardware ingestion stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
