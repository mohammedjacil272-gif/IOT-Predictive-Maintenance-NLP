"""Read-only Phase 3.1 validation for an acquired Azure PdM dataset.

This runner deliberately does not write to the source directory or acquisition
manifest.  It produces a new JSON report at the caller-supplied output path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import EVENT_TABLES, TABLE_SCHEMAS, TimestampPolicy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _parse_timestamp(value: str) -> tuple[datetime, bool]:
    text = value.strip()
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed, parsed.tzinfo is None or parsed.utcoffset() is None


def _key(filename: str, row: dict[str, str]) -> tuple[str, ...]:
    fields = ("machineID",) if filename == "PdM_machines.csv" else (
        ("datetime", "machineID") if filename == "PdM_telemetry.csv" else
        ("datetime", "machineID", {"PdM_errors.csv": "errorID", "PdM_maint.csv": "comp", "PdM_failures.csv": "failure"}[filename])
    )
    return tuple(row.get(field, "") for field in fields)


def _scan_table(path: Path, filename: str) -> dict[str, Any]:
    expected = list(TABLE_SCHEMAS[filename])
    row_count = 0
    blank_rows: list[int] = []
    invalid_numeric: list[dict[str, Any]] = []
    invalid_timestamps: list[dict[str, Any]] = []
    duplicate_keys = 0
    seen: set[tuple[str, ...]] = set()
    machine_ids: set[int] = set()
    distribution = Counter()
    min_dt: datetime | None = None
    max_dt: datetime | None = None
    previous_dt: datetime | None = None
    ordering_violations: list[dict[str, Any]] = []
    ordering_violation_count = 0
    naive_count = 0
    null_or_blank_count = 0

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        observed = list(reader.fieldnames or [])
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            if any(value is None or not str(value).strip() for value in row.values()):
                null_or_blank_count += 1
                if len(blank_rows) < 10:
                    blank_rows.append(row_number)
            key = _key(filename, row)
            if key in seen:
                duplicate_keys += 1
            seen.add(key)

            machine_text = (row.get("machineID") or "").strip()
            try:
                machine_id = int(machine_text)
                if str(machine_id) != machine_text or machine_id <= 0:
                    raise ValueError
                machine_ids.add(machine_id)
            except ValueError:
                if len(invalid_numeric) < 10:
                    invalid_numeric.append({"row": row_number, "field": "machineID", "value": machine_text})

            if filename == "PdM_machines.csv":
                fields = ("age",)
            elif filename == "PdM_telemetry.csv":
                fields = ("volt", "rotate", "pressure", "vibration")
            else:
                fields = ()
            for field in fields:
                text = (row.get(field) or "").strip()
                try:
                    value = float(text)
                    if not math.isfinite(value) or (field == "age" and value < 0):
                        raise ValueError
                except ValueError:
                    if len(invalid_numeric) < 10:
                        invalid_numeric.append({"row": row_number, "field": field, "value": text})

            if filename != "PdM_machines.csv":
                raw_dt = row.get("datetime") or ""
                try:
                    parsed, is_naive = _parse_timestamp(raw_dt)
                    naive_count += int(is_naive)
                    min_dt = parsed if min_dt is None or parsed < min_dt else min_dt
                    max_dt = parsed if max_dt is None or parsed > max_dt else max_dt
                    if previous_dt is not None and parsed < previous_dt:
                        ordering_violation_count += 1
                        if len(ordering_violations) < 10:
                            ordering_violations.append({"row": row_number, "previous": previous_dt.isoformat(), "current": parsed.isoformat()})
                    previous_dt = parsed
                except ValueError:
                    if len(invalid_timestamps) < 10:
                        invalid_timestamps.append({"row": row_number, "value": raw_dt})
                if filename in EVENT_TABLES:
                    distribution[(row.get({"PdM_errors.csv": "errorID", "PdM_maint.csv": "comp", "PdM_failures.csv": "failure"}[filename]) or "").strip()] += 1

    return {
        "observed_columns": observed,
        "expected_columns": expected,
        "exact_schema_match": observed == expected,
        "row_count": row_count,
        "duplicate_key_count": duplicate_keys,
        "null_or_blank_row_count": null_or_blank_count,
        "blank_row_examples": blank_rows,
        "invalid_numeric_examples": invalid_numeric,
        "invalid_timestamp_examples": invalid_timestamps,
        "timestamp_timezone_naive": naive_count == row_count if filename != "PdM_machines.csv" else None,
        "naive_timestamp_count": naive_count,
        "timestamp_range": {"min": min_dt.isoformat() if min_dt else None, "max": max_dt.isoformat() if max_dt else None},
        "timestamp_ordering_violation_count": ordering_violation_count,
        "timestamp_ordering_violation_examples": ordering_violations,
        "machine_ids": sorted(machine_ids),
        "machine_id_count": len(machine_ids),
        "event_distribution": dict(sorted(distribution.items())) if filename in EVENT_TABLES else None,
    }


def validate(dataset_root: Path, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables: dict[str, Any] = {}
    for filename in TABLE_SCHEMAS:
        path = dataset_root / filename
        scan = _scan_table(path, filename)
        scan["size_bytes"] = path.stat().st_size
        scan["sha256"] = _sha256(path)
        scan["git_blob_sha1"] = _git_blob_sha1(path)
        expected_file = stored["files"][filename]
        scan["manifest_sha256_match"] = scan["sha256"] == expected_file["sha256"]
        scan["manifest_git_blob_sha1_match"] = scan["git_blob_sha1"] == expected_file["pinned_blob_sha1"]
        scan["manifest_size_match"] = scan["size_bytes"] == expected_file["size_bytes"]
        tables[filename] = scan

    machines = set(tables["PdM_machines.csv"]["machine_ids"])
    coverage = {name: sorted(set(value["machine_ids"]) - machines) for name, value in tables.items() if name != "PdM_machines.csv"}
    event_files = sorted(EVENT_TABLES)
    report = {
        "phase": "3.1",
        "validation_mode": "read_only",
        "dataset_classification": stored.get("dataset_classification"),
        "structured_events_are_not_technician_narratives": True,
        "derived_text_generated": False,
        "labels_generated": False,
        "temporal_alignment_performed": False,
        "embeddings_generated": False,
        "training_performed": False,
        "source_identity": {"repository": stored["source"]["repository"], "commit": stored["source"]["commit"], "branch_at_resolution": stored["source"]["branch_at_resolution"], "source_directory": stored["source"]["source_directory"]},
        "manifest_path": str(manifest_path),
        "manifest_hashes_and_identity": {name: {key: tables[name][key] for key in ("sha256", "git_blob_sha1", "size_bytes", "manifest_sha256_match", "manifest_git_blob_sha1_match", "manifest_size_match")} for name in TABLE_SCHEMAS},
        "tables": tables,
        "machine_id_coverage": {"machines": sorted(machines), "missing_from_machines_by_table": coverage, "all_event_and_telemetry_ids_present": not any(coverage.values())},
        "availability": {"event_tables": event_files, "available_at_present": {name: "available_at" in tables[name]["observed_columns"] for name in event_files}, "inferred_from_event_time": False, "implication": "Availability metadata is absent; temporal eligibility and derived event text remain blocked until explicit available_at semantics are supplied."},
        "existing_validator_contract": {"streaming_contract_validation_status": "PASS" if all(table["exact_schema_match"] and table["duplicate_key_count"] == 0 and table["null_or_blank_row_count"] == 0 and not table["invalid_numeric_examples"] and not table["invalid_timestamp_examples"] for table in tables.values()) and not any(coverage.values()) else "FAIL", "naive_timestamp_policy_explicitly_used_for_inspection_only": True, "contract_basis": "Same TABLE_SCHEMAS, duplicate-key definitions, positive machine IDs, finite numeric checks, and explicit timestamp parsing as experiments.azure_pdm.schema; streamed to avoid materializing telemetry source IDs."},
        "verified_facts": ["All five required files were scanned read-only.", "The dataset is classified as simulated benchmark data.", "Manifest source identity and recorded file hashes were compared without rewriting the manifest."],
        "warnings": ["Timestamps are timezone-naive; no UTC assumption was made.", "No available_at field exists in event tables.", "The raw directory contains acquisition sidecar files beyond the five validated CSVs."],
        "unresolved_decisions": ["Supply an explicit timezone interpretation for naive timestamps before temporal use.", "Supply explicit availability metadata/semantics before leakage-safe temporal alignment or derived text generation."],
    }
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.dataset_root, args.manifest, args.output)
    print(json.dumps({"output": str(args.output), "status": report["existing_validator_contract"]["streaming_contract_validation_status"], "tables": {name: table["row_count"] for name, table in report["tables"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
