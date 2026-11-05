"""Manifest generation for the synthetic Azure PdM dataset contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from .schema import APPROVED_SYNTHETIC_NAMESPACE, EVENT_TABLES, OPTIONAL_COLUMNS, TABLE_SCHEMAS, AzurePdMValidationError, TimestampPolicy, normalize_rows, read_table
from .temporal import validate_event_availability


class ManifestStaleError(AzurePdMValidationError):
    """Raised when a manifest no longer describes the supplied source files or policy."""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def build_manifest(
    dataset_root: str | Path,
    *,
    timestamp_policy: TimestampPolicy,
    dataset_namespace: str = "azure_pdm_synthetic_fixture",
    is_synthetic: bool = True,
    require_availability: bool = False,
) -> dict[str, Any]:
    """Validate all five tables and return a serializable manifest."""
    root = Path(dataset_root)
    tables: dict[str, Any] = {}
    issues: list[str] = []
    machine_ids_by_table: dict[str, set[int]] = {}

    for filename, required_columns in TABLE_SCHEMAS.items():
        path = root / filename
        table: dict[str, Any] = {
            "path": str(path),
            "required_columns": list(required_columns),
            "observed_columns": [],
            "row_count": 0,
            "sha256": None,
            "date_range_utc": {"min": None, "max": None},
            "status": "FAIL",
        }
        if not path.is_file():
            issues.append(f"Missing required file: {filename}")
            tables[filename] = table
            continue
        try:
            raw_rows, columns = read_table(path)
            table["observed_columns"] = columns
            table["row_count"] = len(raw_rows)
            table["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            optional_columns = OPTIONAL_COLUMNS.get(filename, ())
            allowed_columns = {
                required_columns + combination
                for count in range(len(optional_columns) + 1)
                for combination in combinations(optional_columns, count)
            }
            if tuple(columns) not in allowed_columns:
                raise AzurePdMValidationError(
                    f"{filename}: columns must be exactly {list(required_columns)!r}; got {columns!r}"
                )
            if require_availability and filename in EVENT_TABLES and "available_at" not in columns:
                raise AzurePdMValidationError(
                    f"{filename}: available_at is required by the selected availability policy"
                )
            normalized = normalize_rows(
                filename, raw_rows, timestamp_policy=timestamp_policy
            )
            if require_availability and filename in EVENT_TABLES:
                validate_event_availability(
                    normalized, timestamp_policy=timestamp_policy, require_availability=True
                )
            machine_ids_by_table[filename] = {int(row["machineID"]) for row in normalized}
            dates = [row.get("event_time", row.get("datetime")) for row in normalized]
            dates = [date for date in dates if date is not None]
            table["date_range_utc"] = {
                "min": _iso(min(dates) if dates else None),
                "max": _iso(max(dates) if dates else None),
            }
            table["status"] = "PASS"
        except (AzurePdMValidationError, OSError) as exc:
            issues.append(str(exc))
        tables[filename] = table

    machines = machine_ids_by_table.get("PdM_machines.csv", set())
    for filename, machine_ids in machine_ids_by_table.items():
        if filename == "PdM_machines.csv":
            continue
        missing = sorted(machine_ids - machines)
        if missing:
            issues.append(f"{filename}: machine IDs missing from PdM_machines.csv: {missing}")

    if dataset_namespace != APPROVED_SYNTHETIC_NAMESPACE:
        issues.append(f"Unsupported synthetic namespace: {dataset_namespace!r}")
    if is_synthetic is not True:
        issues.append("This Phase-2 validator requires is_synthetic=true")

    return {
        "manifest_version": "azure_pdm_manifest_v1",
        "dataset_namespace": dataset_namespace,
        "is_synthetic": is_synthetic,
        "source_kind": "local_synthetic_fixture" if is_synthetic else "unapproved",
        "source_record_id_rule": "owner-provided source_record_id, otherwise derived:<filename>:<SHA-256 of canonical row fields>",
        "timestamp_policy": timestamp_policy.describe(),
        "availability_policy": {
            "source_field": "available_at",
            "required_for_real_data": True,
            "required_in_this_manifest": require_availability,
            "event_time_is_not_availability": True,
        },
        "tables": tables,
        "machine_id_coverage": {
            "machines": sorted(machines),
            "all_event_and_telemetry_ids_present": not any(
                set(ids) - machines
                for name, ids in machine_ids_by_table.items()
                if name != "PdM_machines.csv"
            ),
        },
        "validation_status": "PASS" if not issues else "FAIL",
        "validation_issues": issues,
        "training_performed": False,
        "predictive_performance_claim": False,
    }


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Manifest already exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    timestamp_policy: TimestampPolicy,
    require_availability: bool = False,
) -> dict[str, Any]:
    """Recompute and compare a manifest without writing or rebuilding it."""
    path = Path(manifest_path)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestStaleError(f"Unable to read manifest: {path}") from exc
    if stored.get("dataset_namespace") != APPROVED_SYNTHETIC_NAMESPACE:
        raise ManifestStaleError("Manifest namespace is not the approved synthetic namespace")
    if stored.get("is_synthetic") is not True:
        raise ManifestStaleError("Manifest is not explicitly synthetic")
    if stored.get("timestamp_policy") != timestamp_policy.describe():
        raise ManifestStaleError("Manifest timestamp policy differs from the requested policy")
    current = build_manifest(
        dataset_root,
        timestamp_policy=timestamp_policy,
        dataset_namespace=APPROVED_SYNTHETIC_NAMESPACE,
        is_synthetic=True,
        require_availability=require_availability,
    )
    if current["validation_status"] != "PASS":
        raise ManifestStaleError(f"Current source files no longer validate: {current['validation_issues']}")
    for key in ("manifest_version", "source_record_id_rule", "availability_policy", "tables", "machine_id_coverage"):
        if stored.get(key) != current.get(key):
            raise ManifestStaleError(f"Manifest field is stale or changed: {key}")
    return current
