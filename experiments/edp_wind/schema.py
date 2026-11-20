"""Schema discovery and read-only audit for the EDP wind-farm workbooks.

The actual workbook is intentionally not bundled. The adapter records what is
present and fails closed when the Excel engine is unavailable; it never invents
timestamps, identifiers, labels, or text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable


class EDPValidationError(ValueError):
    """Raised when an EDP source cannot be audited safely."""


@dataclass(frozen=True)
class EDPSourceMetadata:
    dataset_id: str = "edp_wind_portugal"
    dataset_version: str = "mendeley-v1"
    source_url: str = "https://data.mendeley.com/datasets/zjxjnjp3xs/1"
    license_name: str = "CC BY 4.0"


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise EDPValidationError(f"Expected a regular non-symlink file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def classify_sheet(sheet_name: str, columns: list[object]) -> str:
    text = " ".join([sheet_name, *[str(column) for column in columns]]).lower()
    if "failure" in text or "fail" in text:
        return "failure"
    if "event" in text or "alarm" in text or "remark" in text:
        return "event"
    if "met" in text or "mast" in text:
        return "met_mast"
    if "scada" in text or "turbine" in text:
        return "scada"
    return "unknown"


def _candidate_columns(columns: list[str], patterns: tuple[str, ...]) -> list[str]:
    normalized = [(column, normalized_column(column)) for column in columns]
    return [column for column, value in normalized if any(pattern in value for pattern in patterns)]


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return value if isinstance(value, (str, int, float, bool, list, dict)) else str(value)


def audit_workbook(
    path: str | Path,
    *,
    metadata: EDPSourceMetadata | None = None,
    excel_file_factory: Callable[..., Any] | None = None,
    read_excel_fn: Callable[..., Any] | None = None,
    sample_rows: int = 3,
) -> dict[str, Any]:
    """Audit workbook structure, types, missing values, and samples read-only."""
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise EDPValidationError(f"Workbook is missing or unsafe: {candidate}")
    if sample_rows < 0:
        raise EDPValidationError("sample_rows must be non-negative")
    try:
        import pandas as pd
    except ImportError as exc:
        raise EDPValidationError("pandas is required for EDP workbook inspection") from exc
    excel_file_factory = excel_file_factory or pd.ExcelFile
    read_excel_fn = read_excel_fn or pd.read_excel
    try:
        workbook = excel_file_factory(candidate)
        sheet_names = list(workbook.sheet_names)
    except Exception as exc:
        raise EDPValidationError(
            "Unable to open workbook; install a compatible Excel engine such as openpyxl: "
            f"{exc}"
        ) from exc
    sheets: list[dict[str, Any]] = []
    for sheet_name in sheet_names:
        try:
            frame = read_excel_fn(candidate, sheet_name=sheet_name)
            columns = [str(column) for column in frame.columns]
            missing = {
                str(column): int(value)
                for column, value in frame.isna().sum().items()
                if int(value) > 0
            }
            dtypes = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
            duplicate_rows = int(frame.duplicated(keep=False).sum())
            candidate_ids = _candidate_columns(
                columns, ("turbine", "machine", "unit", "asset", "equipment", "id")
            )
            candidate_timestamps = _candidate_columns(
                columns, ("timestamp", "datetime", "date", "time", "created", "occurred")
            )
            candidate_text = _candidate_columns(
                columns, ("remark", "description", "comment", "text", "alarm", "event", "failure")
            )
            samples = [
                {str(key): _json_value(value) for key, value in row.items()}
                for row in frame.head(sample_rows).to_dict(orient="records")
            ]
            sheets.append({
                "name": sheet_name,
                "kind": classify_sheet(sheet_name, columns),
                "rows": int(len(frame)),
                "columns": columns,
                "normalized_columns": [normalized_column(column) for column in columns],
                "dtypes": dtypes,
                "missing_values": missing,
                "duplicate_row_count": duplicate_rows,
                "candidate_identifier_columns": candidate_ids,
                "candidate_timestamp_columns": candidate_timestamps,
                "candidate_text_columns": candidate_text,
                "sample_rows": samples,
            })
        except Exception as exc:
            raise EDPValidationError(f"Unable to inspect sheet {sheet_name!r}: {exc}") from exc
    source = metadata or EDPSourceMetadata()
    stat = candidate.stat()
    relationships = []
    for left_index, left in enumerate(sheets):
        left_columns = set(left["normalized_columns"])
        for right in sheets[left_index + 1:]:
            common = sorted(left_columns & set(right["normalized_columns"]))
            if common:
                relationships.append({"left": left["name"], "right": right["name"], "common_columns": common})
    return {
        "audit_generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source.__dict__,
        "file": {
            "name": candidate.name,
            "bytes": stat.st_size,
            "sha256": sha256_file(candidate),
        },
        "sheet_count": len(sheets),
        "sheets": sheets,
        "relationship_candidates": relationships,
        "policy": {
            "raw_immutable": True,
            "timestamps_invented": False,
            "identifiers_invented": False,
            "labels_invented": False,
        },
    }


def audit_directory(root: str | Path, *, metadata: EDPSourceMetadata | None = None) -> dict[str, Any]:
    base = Path(root)
    if not base.is_dir():
        raise EDPValidationError(f"Raw EDP directory is missing: {base}")
    workbooks = sorted(path for path in base.rglob("*.xlsx") if not path.is_symlink())
    if not workbooks:
        raise EDPValidationError(f"No .xlsx workbooks found under {base}")
    return {
        "source": (metadata or EDPSourceMetadata()).__dict__,
        "root": str(base.resolve()),
        "workbooks": [audit_workbook(path, metadata=metadata) for path in workbooks],
    }


def write_report(report: dict[str, Any], destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
