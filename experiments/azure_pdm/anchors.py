"""Read-only telemetry audit and actual-observation anchor generation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


TELEMETRY_FIELDS = ("volt", "rotate", "pressure", "vibration")


def parse_naive_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        raise ValueError("Azure benchmark timestamps must remain timezone-naive")
    return parsed.replace(tzinfo=None)


@dataclass(frozen=True)
class TelemetryAudit:
    rows_by_machine: dict[int, dict[datetime, tuple[float, float, float, float]]]
    per_machine: dict[int, dict[str, Any]]
    total_rows: int
    global_out_of_order_rows: int
    duplicate_keys: int
    invalid_rows: list[dict[str, Any]]
    start: datetime
    end: datetime


def read_and_audit_telemetry(path: Path) -> TelemetryAudit:
    rows_by_machine: dict[int, dict[datetime, tuple[float, float, float, float]]] = defaultdict(dict)
    per_machine: dict[int, dict[str, Any]] = {}
    invalid_rows: list[dict[str, Any]] = []
    total_rows = 0
    duplicate_keys = 0
    global_out_of_order = 0
    previous_by_machine: dict[int, datetime] = {}

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = ["datetime", "machineID", *TELEMETRY_FIELDS]
        if list(reader.fieldnames or []) != expected:
            raise ValueError(f"Telemetry schema mismatch: {reader.fieldnames!r}")
        for row_number, raw in enumerate(reader, start=2):
            total_rows += 1
            try:
                machine = int((raw["machineID"] or "").strip())
                timestamp = parse_naive_timestamp(raw["datetime"] or "")
                values = tuple(float((raw[field] or "").strip()) for field in TELEMETRY_FIELDS)
                if machine <= 0 or not all(math.isfinite(value) for value in values):
                    raise ValueError("machineID must be positive and telemetry values finite")
            except (TypeError, ValueError) as exc:
                invalid_rows.append({"row": row_number, "error": str(exc)})
                continue
            if machine in previous_by_machine and timestamp < previous_by_machine[machine]:
                global_out_of_order += 1
            previous_by_machine[machine] = timestamp
            machine_rows = rows_by_machine[machine]
            if timestamp in machine_rows:
                duplicate_keys += 1
            else:
                machine_rows[timestamp] = values

    if not rows_by_machine:
        raise ValueError("No valid telemetry rows found")

    all_timestamps = [timestamp for rows in rows_by_machine.values() for timestamp in rows]
    for machine, rows in sorted(rows_by_machine.items()):
        ordered = sorted(rows)
        start, end = ordered[0], ordered[-1]
        expected_count = int((end - start).total_seconds() // 3600) + 1
        expected_timestamps = {start + timedelta(hours=i) for i in range(expected_count)}
        missing = sorted(expected_timestamps - set(rows))
        per_machine[machine] = {
            "row_count": len(rows),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "expected_hourly_count": expected_count,
            "missing_hourly_count": len(missing),
            "missing_hourly_examples": [value.isoformat() for value in missing[:20]],
            "complete_hourly_coverage": not missing,
            "source_order_out_of_order_count": sum(
                1 for left, right in zip(
                    [timestamp for timestamp in rows],
                    [timestamp for timestamp in rows][1:],
                ) if right < left
            ),
        }
    return TelemetryAudit(
        rows_by_machine=dict(rows_by_machine),
        per_machine=per_machine,
        total_rows=total_rows,
        global_out_of_order_rows=global_out_of_order,
        duplicate_keys=duplicate_keys,
        invalid_rows=invalid_rows,
        start=min(all_timestamps),
        end=max(all_timestamps),
    )


def actual_observation_anchors(
    rows: dict[datetime, tuple[float, float, float, float]],
    *,
    cadence_hours: int = 3,
) -> list[datetime]:
    """Return actual observations on a machine-relative 3-hour cadence."""
    if cadence_hours <= 0:
        raise ValueError("cadence_hours must be positive")
    ordered = sorted(rows)
    start = ordered[0]
    cadence_seconds = cadence_hours * 3600
    return [
        timestamp
        for timestamp in ordered
        if int((timestamp - start).total_seconds()) % cadence_seconds == 0
    ]
