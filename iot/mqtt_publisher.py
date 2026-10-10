"""Publish validated FD001 training sensor replay messages to MQTT."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Protocol, Sequence, TextIO

from .local_replay import DEFAULT_TRAIN_PATH, iter_fd001_training_messages
from .sensor_message import SensorMessage, validate_sensor_message


DEFAULT_MQTT_HOST = "localhost"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_TOPIC = "industrial/sensors"
DEFAULT_MQTT_QOS = 0

ENV_MQTT_HOST = "IOT_MQTT_HOST"
ENV_MQTT_PORT = "IOT_MQTT_PORT"
ENV_MQTT_TOPIC = "IOT_MQTT_TOPIC"
ENV_MQTT_QOS = "IOT_MQTT_QOS"
ENV_REPLAY_LIMIT = "IOT_REPLAY_LIMIT"
ENV_TRAIN_PATH = "IOT_TRAIN_PATH"


class MqttPublisherError(RuntimeError):
    """Raised when MQTT publishing cannot be completed."""


class MqttClient(Protocol):
    def connect(self, host: str, port: int) -> int:
        """Connect to an MQTT broker."""

    def publish(self, topic: str, payload: str, qos: int):
        """Publish one payload."""

    def disconnect(self) -> int:
        """Disconnect from an MQTT broker."""


@dataclass(frozen=True)
class MqttPublisherConfig:
    host: str = DEFAULT_MQTT_HOST
    port: int = DEFAULT_MQTT_PORT
    topic: str = DEFAULT_MQTT_TOPIC
    qos: int = DEFAULT_MQTT_QOS
    train_path: Path = DEFAULT_TRAIN_PATH
    limit: int | None = None
    dry_run: bool = False


def serialize_sensor_message(message: SensorMessage) -> str:
    """Serialize a validated sensor message as compact JSON."""

    normalized = validate_sensor_message(message.to_dict())
    return json.dumps(normalized.to_dict(), separators=(",", ":"), sort_keys=False)


def publish_replay(
    config: MqttPublisherConfig,
    *,
    client_factory: Callable[[], MqttClient] | None = None,
    output: TextIO | None = None,
) -> int:
    """Publish FD001 training replay messages and return the count published."""

    _validate_config(config)
    stream = output if output is not None else sys.stdout
    published = 0

    if config.dry_run:
        for message in iter_fd001_training_messages(config.train_path, limit=config.limit):
            payload = serialize_sensor_message(message)
            print(f"DRY-RUN topic={config.topic} qos={config.qos} payload={payload}", file=stream)
            published += 1
        return published

    factory = client_factory or _create_paho_client
    client = factory()
    try:
        _connect_client(client, config)
        for message in iter_fd001_training_messages(config.train_path, limit=config.limit):
            payload = serialize_sensor_message(message)
            _publish_message(client, config.topic, payload, config.qos)
            published += 1
    finally:
        _disconnect_client(client)

    return published


def parse_config(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> MqttPublisherConfig:
    """Build publisher config from command-line arguments and environment."""

    source_env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Publish FD001 training sensor replay messages to MQTT.")
    parser.add_argument("--host", default=source_env.get(ENV_MQTT_HOST, DEFAULT_MQTT_HOST))
    parser.add_argument("--port", type=int, default=_env_int(source_env, ENV_MQTT_PORT, DEFAULT_MQTT_PORT))
    parser.add_argument("--topic", default=source_env.get(ENV_MQTT_TOPIC, DEFAULT_MQTT_TOPIC))
    parser.add_argument("--qos", type=int, default=_env_int(source_env, ENV_MQTT_QOS, DEFAULT_MQTT_QOS))
    parser.add_argument("--train-path", default=source_env.get(ENV_TRAIN_PATH, str(DEFAULT_TRAIN_PATH)))
    parser.add_argument("--limit", type=int, default=_env_optional_int(source_env, ENV_REPLAY_LIMIT))
    parser.add_argument("--dry-run", action="store_true", help="Serialize messages without connecting to a broker.")
    args = parser.parse_args(argv)

    config = MqttPublisherConfig(
        host=args.host,
        port=args.port,
        topic=args.topic,
        qos=args.qos,
        train_path=Path(args.train_path),
        limit=args.limit,
        dry_run=args.dry_run,
    )
    _validate_config(config)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_config(argv)
        count = publish_replay(config)
    except (MqttPublisherError, OSError, ValueError) as exc:
        print(f"MQTT publisher error: {exc}", file=sys.stderr)
        return 1

    mode = "dry-run" if config.dry_run else "published"
    print(f"MQTT sensor publisher {mode} messages: {count}")
    return 0


def _create_paho_client() -> MqttClient:
    try:
        mqtt = importlib.import_module("paho.mqtt.client")
    except ImportError as exc:
        raise MqttPublisherError(
            "Missing Paho MQTT dependency. Install it with: python -m pip install paho-mqtt"
        ) from exc

    return mqtt.Client()


def _connect_client(client: MqttClient, config: MqttPublisherConfig) -> None:
    try:
        result = client.connect(config.host, config.port)
    except Exception as exc:  # noqa: BLE001 - surface third-party client failures clearly.
        raise MqttPublisherError(f"Failed to connect to MQTT broker {config.host}:{config.port}: {exc}") from exc

    rc = _result_code(result)
    if rc != 0:
        raise MqttPublisherError(f"Failed to connect to MQTT broker {config.host}:{config.port}; return code {rc}.")


def _publish_message(client: MqttClient, topic: str, payload: str, qos: int) -> None:
    try:
        result = client.publish(topic, payload, qos=qos)
        wait_for_publish = getattr(result, "wait_for_publish", None)
        if callable(wait_for_publish):
            wait_for_publish()
    except Exception as exc:  # noqa: BLE001 - surface third-party client failures clearly.
        raise MqttPublisherError(f"Failed to publish MQTT message to {topic}: {exc}") from exc

    rc = _result_code(result)
    if rc != 0:
        raise MqttPublisherError(f"Failed to publish MQTT message to {topic}; return code {rc}.")


def _disconnect_client(client: MqttClient) -> None:
    disconnect = getattr(client, "disconnect", None)
    if callable(disconnect):
        disconnect()


def _result_code(result: object) -> int:
    if isinstance(result, int):
        return result
    rc = getattr(result, "rc", None)
    if isinstance(rc, int):
        return rc
    return 0


def _validate_config(config: MqttPublisherConfig) -> None:
    if not config.host:
        raise MqttPublisherError("MQTT broker host must not be empty.")
    if not 1 <= config.port <= 65535:
        raise MqttPublisherError("MQTT broker port must be between 1 and 65535.")
    if not config.topic:
        raise MqttPublisherError("MQTT topic must not be empty.")
    if config.qos not in (0, 1, 2):
        raise MqttPublisherError("MQTT QoS must be 0, 1, or 2.")
    if config.limit is not None and config.limit < 0:
        raise MqttPublisherError("Replay limit must be non-negative.")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise MqttPublisherError(f"{name} must be an integer.") from exc


def _env_optional_int(env: Mapping[str, str], name: str) -> int | None:
    raw_value = env.get(name)
    if raw_value is None or raw_value == "":
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise MqttPublisherError(f"{name} must be an integer.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
