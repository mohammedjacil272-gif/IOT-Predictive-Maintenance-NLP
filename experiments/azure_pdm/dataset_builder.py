"""Deterministic temporal index/target dataset builder for Azure PdM."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .anchors import TelemetryAudit, actual_observation_anchors, read_and_audit_telemetry
from .targets import TARGET_CLASSES, FailureAudit, read_and_audit_failures, target_for_anchor
from .windows import LookbackWindow
from .splits import SplitPolicy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_tuple(value: object) -> str:
    return json.dumps(list(value) if isinstance(value, tuple) else value, separators=(",", ":"), sort_keys=True)


def _window_complete(rows: dict[datetime, tuple[float, float, float, float]], window: LookbackWindow, anchor: datetime) -> bool:
    return all(timestamp in rows for timestamp in window.expected_hourly_timestamps(anchor))


def build_temporal_dataset(
    *,
    dataset_root: Path,
    output_root: Path,
    contract_version: str = "azure_pdm_experiment_contract_v1.1",
) -> dict[str, Any]:
    """Build a temporal index with target provenance, never feature values."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite derived output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    telemetry_path = dataset_root / "PdM_telemetry.csv"
    failures_path = dataset_root / "PdM_failures.csv"
    telemetry = read_and_audit_telemetry(telemetry_path)
    failures = read_and_audit_failures(failures_path)
    split_policy = SplitPolicy(telemetry.start, telemetry.end)
    windows = (LookbackWindow(3), LookbackWindow(24))

    rows: list[dict[str, Any]] = []
    exclusion_counts = Counter()
    anchor_counts = Counter()
    target_counts = Counter()
    per_machine_output = Counter()
    for machine_id, machine_rows in sorted(telemetry.rows_by_machine.items()):
        events = failures.events_by_machine.get(machine_id, tuple())
        for anchor in actual_observation_anchors(machine_rows, cadence_hours=3):
            anchor_counts["actual_observation_anchors"] += 1
            split, split_reason = split_policy.assign(anchor)
            if split is None:
                exclusion_counts[split_reason or "unassigned_split"] += 1
                continue
            if anchor + timedelta(hours=24) > telemetry.end:
                exclusion_counts["incomplete_24h_horizon"] += 1
                continue
            target = target_for_anchor(events, anchor)
            target_counts[(split, target["target_label"])] += 1
            for window in windows:
                anchor_counts[f"candidate_anchors_lb_{window.hours}h"] += 1
                if not _window_complete(machine_rows, window, anchor):
                    exclusion_counts[f"missing_or_incomplete_{window.hours}h_lookback"] += 1
                    continue
                sample_id = f"azure_pdm_v1__m{machine_id:03d}__t{anchor.strftime('%Y%m%dT%H%M%S')}__lb{window.hours}h"
                lookback_start, lookback_end = window.bounds(anchor)
                row = {
                    "sample_id": sample_id,
                    "machineID": machine_id,
                    "anchor_timestamp": anchor.isoformat(sep=" "),
                    "lookback_hours": window.hours,
                    "lookback_start_exclusive": lookback_start.isoformat(sep=" "),
                    "lookback_end_exclusive": lookback_end.isoformat(sep=" "),
                    "telemetry_observation_count": window.hours,
                    "split": split,
                    "target_label": target["target_label"],
                    "target_components": _json_tuple(target["target_components"]),
                    "target_event_refs": _json_tuple(target["target_event_refs"]),
                    "target_event_timestamps": _json_tuple(tuple(
                        event.timestamp.isoformat(sep=" ")
                        for event in events
                        if event.source_ref in target["target_event_refs"]
                    )),
                    "target_event_count": target["target_event_count"],
                    "target_is_recorded_evidence_only": True,
                    "telemetry_source_file": "PdM_telemetry.csv",
                    "failure_source_file": "PdM_failures.csv",
                    "contract_version": contract_version,
                }
                rows.append(row)
                per_machine_output[(machine_id, window.hours)] += 1

    rows.sort(key=lambda row: (row["machineID"], row["anchor_timestamp"], row["lookback_hours"]))
    fields = list(rows[0]) if rows else [
        "sample_id", "machineID", "anchor_timestamp", "lookback_hours",
        "lookback_start_exclusive", "lookback_end_exclusive", "telemetry_observation_count",
        "split", "target_label", "target_components", "target_event_refs", "target_event_timestamps",
        "target_event_count", "target_is_recorded_evidence_only", "telemetry_source_file",
        "failure_source_file", "contract_version",
    ]
    dataset_path = output_root / "phase4_temporal_dataset.csv"
    with dataset_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(row["split"] for row in rows)
    class_counts = Counter(row["target_label"] for row in rows)
    split_class_counts = Counter((row["split"], row["target_label"]) for row in rows)
    failure_component_counts = Counter(
        event.component for events in failures.events_by_machine.values() for event in events
    )
    failure_machine_component_counts = {
        str(machine): dict(Counter(event.component for event in events))
        for machine, events in sorted(failures.events_by_machine.items())
    }
    audit = {
        "contract_version": contract_version,
        "generation_mode": "read_only_source_scan_and_deterministic_temporal_index_generation",
        "telemetry": {
            "total_rows": telemetry.total_rows,
            "machine_count": len(telemetry.rows_by_machine),
            "global_out_of_order_rows": telemetry.global_out_of_order_rows,
            "duplicate_keys": telemetry.duplicate_keys,
            "invalid_rows": telemetry.invalid_rows,
            "coverage_start": telemetry.start.isoformat(),
            "coverage_end": telemetry.end.isoformat(),
            "per_machine": telemetry.per_machine,
        },
        "failures": {
            "row_count": failures.row_count,
            "duplicate_keys": failures.duplicate_keys,
            "invalid_rows": failures.invalid_rows,
            "unsupported_components": failures.unsupported_components,
            "events_by_machine_count": {str(k): len(v) for k, v in sorted(failures.events_by_machine.items())},
            "component_counts": dict(sorted(failure_component_counts.items())),
            "machine_component_counts": failure_machine_component_counts,
        },
        "split_policy": split_policy.describe(),
        "anchor_generation": dict(anchor_counts),
        "exclusion_counts": dict(exclusion_counts),
        "output_row_count": len(rows),
        "output_split_counts": dict(split_counts),
        "output_class_counts": dict(class_counts),
        "output_split_class_counts": {f"{split}:{label}": count for (split, label), count in sorted(split_class_counts.items())},
        "per_machine_output_counts": {f"{machine}:{hours}h": count for (machine, hours), count in sorted(per_machine_output.items())},
        "target_classes": list(TARGET_CLASSES),
        "event_features_included": False,
        "failure_features_included": False,
        "labels_generated": True,
        "features_generated": False,
        "embeddings_generated": False,
        "training_performed": False,
        "inference_performed": False,
    }
    audit_path = output_root / "phase4_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "manifest_version": "azure_pdm_phase4_manifest_v1",
        "contract_version": contract_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_files": {},
        "configuration": {
            "prediction_horizon_hours": 24,
            "cadence_hours": 3,
            "lookback_hours": [3, 24],
            "lookback_interval": "[anchor-hours, anchor)",
            "target_interval": "(anchor, anchor+24h]",
            "timezone_policy": "timezone-naive benchmark timestamps preserved; no UTC/IST assumption",
            "event_availability_policy": "errors and maintenance excluded; availability not inferred",
            "split_policy": split_policy.describe(),
        },
        "output": {"path": str(dataset_path), "sha256": sha256_file(dataset_path), "row_count": len(rows)},
        "audit_path": str(audit_path),
        "audit_summary": {
            "output_split_counts": dict(split_counts),
            "output_class_counts": dict(class_counts),
            "exclusion_counts": dict(exclusion_counts),
        },
    }
    for source_name in ("PdM_telemetry.csv", "PdM_errors.csv", "PdM_maint.csv", "PdM_failures.csv", "PdM_machines.csv"):
        source_path = dataset_root / source_name
        with source_path.open("r", newline="", encoding="utf-8") as source_stream:
            source_row_count = max(0, sum(1 for _ in csv.DictReader(source_stream)))
        manifest["source_files"][source_name] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "row_count": source_row_count,
        }
    manifest_path = output_root / "phase4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dataset_path": dataset_path, "audit_path": audit_path, "manifest_path": manifest_path, "audit": audit, "manifest": manifest}
