"""Offline-first Azure PdM telemetry streaming adapter.

This module accepts telemetry observations only. It deliberately has no MQTT or
InfluxDB connection code: callers may feed records from those systems later,
but the guaranteed demonstration path is deterministic local replay.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from .inference import AzurePdMPredictor, AzureInferenceError

LOGGER = logging.getLogger(__name__)
TELEMETRY_FIELDS = ("volt", "rotate", "pressure", "vibration")
REQUIRED_FIELDS = ("datetime", "machineID", *TELEMETRY_FIELDS)


class AzureStreamError(ValueError):
    """Raised when a streaming record violates the Azure telemetry contract."""


def read_machine_metadata(path: str | Path) -> dict[int, dict[str, Any]]:
    machines: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if set(row) != {"machineID", "model", "age"}:
                raise AzureStreamError("machine metadata schema must be machineID, model, age")
            try:
                machine_id = int(row["machineID"]); age = float(row["age"])
            except (TypeError, ValueError) as exc:
                raise AzureStreamError("machine metadata contains an invalid ID or age") from exc
            if machine_id <= 0 or not row["model"] or age < 0:
                raise AzureStreamError("machine metadata contains an invalid machine")
            machines[machine_id] = {"machine_model": row["model"], "machine_age": age}
    return machines


def iter_telemetry_csv(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise AzureStreamError(f"telemetry schema must be exactly {REQUIRED_FIELDS}")
        for row in reader:
            yield dict(row)


class AzureTelemetryWindowAdapter:
    """Accumulate contiguous hourly records and infer only complete windows."""

    def __init__(self, predictor: AzurePdMPredictor, machines: dict[int, dict[str, Any]], *, lookback_hours: int = 24) -> None:
        if lookback_hours != predictor.artifact["lookback_hours"]:
            raise AzureStreamError("adapter lookback does not match saved model artifact")
        self.predictor = predictor
        self.machines = machines
        self.lookback_hours = lookback_hours
        self.buffers: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=lookback_hours))
        self.last_timestamp: dict[int, datetime] = {}
        self.seen_keys: set[tuple[int, datetime]] = set()
        self.stats = {"records_seen": 0, "predictions_emitted": 0, "duplicates_rejected": 0, "invalid_records": 0, "gap_resets": 0, "incomplete_windows": 0, "unknown_machines": 0, "latencies_ms": []}

    @staticmethod
    def _validate_record(record: dict[str, Any]) -> tuple[int, datetime, dict[str, Any]]:
        if set(record) != set(REQUIRED_FIELDS):
            raise AzureStreamError(f"telemetry record fields must be exactly {REQUIRED_FIELDS}")
        try:
            machine_id = int(record["machineID"])
            timestamp = datetime.fromisoformat(str(record["datetime"]))
            values = {field: float(record[field]) for field in TELEMETRY_FIELDS}
        except (TypeError, ValueError) as exc:
            raise AzureStreamError("telemetry record contains malformed timestamp, ID, or numeric value") from exc
        if machine_id <= 0 or timestamp.tzinfo is not None:
            raise AzureStreamError("machineID must be positive and timestamps must remain timezone-naive")
        if any(value != value or value in (float("inf"), float("-inf")) for value in values.values()):
            raise AzureStreamError("telemetry values must be finite")
        normalized = {"datetime": timestamp.isoformat(sep=" "), "volt": values["volt"], "rotate": values["rotate"], "pressure": values["pressure"], "vibration": values["vibration"]}
        return machine_id, timestamp, normalized

    def ingest(self, record: dict[str, Any]) -> dict[str, Any] | None:
        self.stats["records_seen"] += 1
        try:
            machine_id, timestamp, normalized = self._validate_record(record)
        except AzureStreamError:
            self.stats["invalid_records"] += 1
            raise
        if machine_id not in self.machines:
            self.stats["unknown_machines"] += 1
            raise AzureStreamError(f"machineID {machine_id} has no approved metadata")
        key = (machine_id, timestamp)
        if key in self.seen_keys:
            self.stats["duplicates_rejected"] += 1
            LOGGER.warning("duplicate telemetry rejected", extra={"machine_id": machine_id, "timestamp": timestamp.isoformat()})
            return None
        self.seen_keys.add(key)
        previous = self.last_timestamp.get(machine_id)
        if previous is not None and timestamp != previous + timedelta(hours=1):
            self.buffers[machine_id].clear()
            self.stats["gap_resets"] += 1
            LOGGER.warning("telemetry gap or out-of-order record reset window", extra={"machine_id": machine_id, "previous": previous.isoformat(), "current": timestamp.isoformat()})
        self.last_timestamp[machine_id] = timestamp
        self.buffers[machine_id].append(normalized)
        if len(self.buffers[machine_id]) < self.lookback_hours:
            self.stats["incomplete_windows"] += 1
            return None
        payload = {"machine_id": machine_id, **self.machines[machine_id], "telemetry": list(self.buffers[machine_id])}
        started = time.perf_counter()
        result = self.predictor.predict_one(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.stats["latencies_ms"].append(latency_ms)
        self.stats["predictions_emitted"] += 1
        result.update({"observation_time": timestamp.isoformat(sep=" "), "benchmark_status": "simulated benchmark; offline replay; not a verified real-world failure", "input_window_start": self.buffers[machine_id][0]["datetime"], "input_window_end": self.buffers[machine_id][-1]["datetime"], "latency_ms": latency_ms})
        return result

    def replay(self, records: Iterable[dict[str, Any]], *, max_predictions: int | None = None) -> Iterator[dict[str, Any]]:
        for record in records:
            result = self.ingest(record)
            if result is not None:
                yield result
                if max_predictions is not None and self.stats["predictions_emitted"] >= max_predictions:
                    return

    def summary(self) -> dict[str, Any]:
        latencies = self.stats["latencies_ms"]
        result = {key: value for key, value in self.stats.items() if key != "latencies_ms"}
        result["latency_ms"] = {"count": len(latencies), "mean": sum(latencies) / len(latencies) if latencies else None, "max": max(latencies) if latencies else None}
        return result


def run_offline_replay(*, telemetry_csv: str | Path, machines_csv: str | Path, model_path: str | Path, output_jsonl: str | Path, max_predictions: int = 10) -> dict[str, Any]:
    predictor = AzurePdMPredictor(model_path)
    machines = read_machine_metadata(machines_csv)
    adapter = AzureTelemetryWindowAdapter(predictor, machines)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emitted: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        for result in adapter.replay(iter_telemetry_csv(telemetry_csv), max_predictions=max_predictions):
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            emitted.append(result)
    return {"output_path": str(output_path), "predictions": emitted, "summary": adapter.summary(), "model_version": predictor.artifact["provenance"]["model_version"], "benchmark_status": "simulated benchmark; offline replay"}
