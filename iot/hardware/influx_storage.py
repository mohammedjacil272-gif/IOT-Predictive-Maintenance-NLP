"""InfluxDB storage adapter for hardware telemetry, separate from FD001."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Mapping, Protocol

from .message import HardwareTelemetry

DEFAULT_HARDWARE_MEASUREMENT = "hardware_sensor_reading"
ENV_INFLUX_URL = "IOT_HARDWARE_INFLUXDB_URL"
ENV_INFLUX_ORG = "IOT_HARDWARE_INFLUXDB_ORG"
ENV_INFLUX_BUCKET = "IOT_HARDWARE_INFLUXDB_BUCKET"
ENV_INFLUX_TOKEN = "IOT_HARDWARE_INFLUXDB_TOKEN"
ENV_INFLUX_MEASUREMENT = "IOT_HARDWARE_INFLUXDB_MEASUREMENT"


class HardwareInfluxError(RuntimeError):
    """Raised when hardware-specific InfluxDB storage is unavailable."""


class WriteApi(Protocol):
    def write(self, *, bucket: str, org: str, record: object) -> None: ...


@dataclass(frozen=True)
class InfluxStorageConfig:
    url: str
    org: str
    bucket: str
    token: str
    measurement: str = DEFAULT_HARDWARE_MEASUREMENT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "InfluxStorageConfig":
        values = os.environ if env is None else env
        config = cls(values.get(ENV_INFLUX_URL, ""), values.get(ENV_INFLUX_ORG, ""),
                     values.get(ENV_INFLUX_BUCKET, ""), values.get(ENV_INFLUX_TOKEN, ""),
                     values.get(ENV_INFLUX_MEASUREMENT, DEFAULT_HARDWARE_MEASUREMENT))
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("url", "org", "bucket", "token", "measurement"):
            if not getattr(self, name):
                raise HardwareInfluxError(f"InfluxDB {name} is required for hardware telemetry.")


def build_point(telemetry: HardwareTelemetry, measurement: str = DEFAULT_HARDWARE_MEASUREMENT) -> object:
    try:
        influx = importlib.import_module("influxdb_client")
    except ImportError as exc:
        raise HardwareInfluxError("Missing InfluxDB dependency. Install influxdb-client.") from exc
    point = influx.Point(measurement)
    for key, value in (("device_id", telemetry.device_id), ("session_id", telemetry.session_id)):
        point = point.tag(key, value)
    for key, value in (
        ("temperature", telemetry.temperature), ("vibration", telemetry.vibration),
        ("rpm", telemetry.rpm), ("pressure", telemetry.pressure),
        ("operating_status", telemetry.operating_status),
        ("sequence_number", telemetry.sequence_number), ("schema_version", telemetry.schema_version),
    ):
        point = point.field(key, value)
    # Keep message precision (up to milliseconds) and use the actual event time.
    return point.time(telemetry.timestamp)


def write_telemetry(telemetry: HardwareTelemetry, config: InfluxStorageConfig,
                    write_api: WriteApi | None = None) -> None:
    config.validate()
    client = None
    try:
        if write_api is None:
            influx = importlib.import_module("influxdb_client")
            write_options = importlib.import_module("influxdb_client.client.write_api")
            client = influx.InfluxDBClient(url=config.url, token=config.token, org=config.org)
            # Synchronous writes surface errors inside the MQTT callback so the
            # sequence tracker only marks messages stored after Influx confirms.
            write_api = client.write_api(write_options=write_options.SYNCHRONOUS)
        point = build_point(telemetry, config.measurement)
        write_api.write(bucket=config.bucket, org=config.org, record=point)
    except HardwareInfluxError:
        raise
    except Exception as exc:
        raise HardwareInfluxError("Failed to write hardware telemetry to InfluxDB.") from exc
    finally:
        if client is not None:
            client.close()
