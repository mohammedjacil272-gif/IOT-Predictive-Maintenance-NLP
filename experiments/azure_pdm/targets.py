"""Fail-closed, machine-specific Azure PdM target construction."""

from __future__ import annotations

import csv
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .anchors import parse_naive_timestamp


TARGET_COMPONENTS = frozenset({"comp1", "comp2", "comp3", "comp4"})
TARGET_CLASSES = ("none", "comp1", "comp2", "comp3", "comp4", "multi_failure")


@dataclass(frozen=True)
class FailureEvent:
    machine_id: int
    timestamp: datetime
    component: str
    source_ref: str


@dataclass(frozen=True)
class FailureAudit:
    events_by_machine: dict[int, tuple[FailureEvent, ...]]
    row_count: int
    duplicate_keys: int
    invalid_rows: list[dict[str, Any]]
    unsupported_components: list[dict[str, Any]]


def read_and_audit_failures(path: Path) -> FailureAudit:
    by_machine: dict[int, list[FailureEvent]] = defaultdict(list)
    seen: set[tuple[int, datetime, str]] = set()
    invalid_rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    duplicate_keys = 0
    row_count = 0
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = ["datetime", "machineID", "failure"]
        if list(reader.fieldnames or []) != expected:
            raise ValueError(f"Failure schema mismatch: {reader.fieldnames!r}")
        for row_number, raw in enumerate(reader, start=2):
            row_count += 1
            try:
                machine = int((raw["machineID"] or "").strip())
                timestamp = parse_naive_timestamp(raw["datetime"] or "")
                component = (raw["failure"] or "").strip()
                if machine <= 0:
                    raise ValueError("machineID must be positive")
            except (TypeError, ValueError) as exc:
                invalid_rows.append({"row": row_number, "error": str(exc)})
                continue
            if component not in TARGET_COMPONENTS:
                unsupported.append({"row": row_number, "value": component})
                continue
            key = (machine, timestamp, component)
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
            by_machine[machine].append(
                FailureEvent(machine, timestamp, component, f"PdM_failures.csv:{row_number}")
            )
    if invalid_rows or unsupported or duplicate_keys:
        raise ValueError(
            "Failure audit failed closed: "
            f"invalid={len(invalid_rows)}, unsupported={len(unsupported)}, duplicates={duplicate_keys}"
        )
    ordered = {machine: tuple(sorted(events, key=lambda event: (event.timestamp, event.component, event.source_ref))) for machine, events in by_machine.items()}
    return FailureAudit(ordered, row_count, duplicate_keys, invalid_rows, unsupported)


def target_for_anchor(
    events: tuple[FailureEvent, ...],
    anchor: datetime,
    *,
    horizon_hours: int = 24,
) -> dict[str, Any]:
    """Build one target from (anchor, anchor+horizon], without global fill."""
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    timestamps = [event.timestamp for event in events]
    start = bisect_right(timestamps, anchor)
    end = bisect_right(timestamps, anchor + timedelta(hours=horizon_hours))
    selected = events[start:end]
    components = tuple(sorted({event.component for event in selected}))
    if not components:
        label = "none"
    elif len(components) == 1:
        label = components[0]
    else:
        label = "multi_failure"
    return {
        "target_label": label,
        "target_components": components,
        "target_event_refs": tuple(event.source_ref for event in selected),
        "target_event_count": len(selected),
    }
