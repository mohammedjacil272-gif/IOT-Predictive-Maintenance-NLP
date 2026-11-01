"""Validation checks for generated Azure PdM temporal datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .splits import SplitPolicy
from .targets import TARGET_CLASSES


def validate_temporal_dataset(path: Path, *, split_policy: SplitPolicy) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    errors: list[str] = []
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id")
    expected_splits = {"train", "validation", "test"}
    seen_by_key: set[tuple[int, str, int]] = set()
    split_counts = Counter()
    class_counts = Counter()
    for row in rows:
        machine = int(row["machineID"])
        anchor = datetime.fromisoformat(row["anchor_timestamp"])
        key = (machine, row["anchor_timestamp"], int(row["lookback_hours"]))
        if key in seen_by_key:
            errors.append(f"duplicate temporal key {key}")
        seen_by_key.add(key)
        split = row["split"]
        if split not in expected_splits:
            errors.append(f"unsupported split {split!r}")
        assigned, reason = split_policy.assign(anchor)
        if assigned != split:
            errors.append(f"split mismatch for {key}: expected {assigned!r}, got {split!r} ({reason})")
        label = row["target_label"]
        if label not in TARGET_CLASSES:
            errors.append(f"unsupported target class {label!r}")
        target_refs = json.loads(row["target_event_refs"])
        target_times = [datetime.fromisoformat(value) for value in json.loads(row["target_event_timestamps"])]
        if int(row["target_event_count"]) != len(target_refs):
            errors.append(f"target event count mismatch for {key}")
        if len(target_refs) != len(target_times):
            errors.append(f"target event timestamp count mismatch for {key}")
        for target_time in target_times:
            if not (anchor < target_time <= anchor + timedelta(hours=24)):
                errors.append(f"target event outside (t,t+24h] for {key}")
            if split == "train" and target_time >= split_policy.first_cutpoint:
                errors.append(f"train target crosses first boundary for {key}")
            if split == "validation" and target_time >= split_policy.second_cutpoint:
                errors.append(f"validation target crosses second boundary for {key}")
        split_counts[split] += 1
        class_counts[label] += 1
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "row_count": len(rows),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
    }
