"""Mechanical intake validation for a Rafaz NLP artifact package.

This module validates files as supplied. It does not turn fixture artifacts into
genuine data, establish authenticity, infer provenance, or connect to training.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import zipfile

import torch


REQUIRED_FILES = (
    "embeddings.pt",
    "embedding_metadata.csv",
    "processed_logs.csv",
    "entities.jsonl",
    "severity.csv",
    "verification_report.json",
)
EMBEDDING_DIM = 768
TIMESTAMP_FIELDS = ("event_timestamp", "created_at", "ingested_at", "available_at")
METADATA_FIELDS = (
    "row_index", "log_id", "unit_id", "cycle", "source_split", "model_name",
    "max_length", "pooling", "embedding_dim",
)
PROVENANCE_FIELDS = ("source_system", "source_record_id", "source_dataset", "source_dataset_version")
REVISION_FIELDS = ("model_name", "tokenizer_name", "model_revision", "tokenizer_revision")


@dataclass(frozen=True)
class ArtifactValidationLimits:
    """Resource bounds applied before and during artifact parsing."""

    max_file_bytes: int = 256 * 1024 * 1024
    max_csv_rows: int = 1_000_000
    max_jsonl_line_bytes: int = 1 * 1024 * 1024
    max_jsonl_records: int = 1_000_000
    max_tensor_elements: int = 100_000_000

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


DEFAULT_LIMITS = ArtifactValidationLimits()


class _ResourceLimitExceeded(ValueError):
    """Raised when an artifact exceeds a configured intake bound."""


class _UnsupportedTensorFormat(ValueError):
    """Raised when safe tensor preflight cannot be performed."""


@contextmanager
def _open_checked(path: Path, *, max_file_bytes: int):
    """Open one regular, non-symlink file and validate its opened handle."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(os.fspath(path), os.O_RDONLY | nofollow)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path.name} is not a regular file")
        if info.st_size > max_file_bytes:
            raise _ResourceLimitExceeded(
                f"{path.name} exceeds maximum file size ({max_file_bytes} bytes)."
            )
        with os.fdopen(fd, "rb") as stream:
            fd = None
            yield stream
    finally:
        if fd is not None:
            os.close(fd)


def _read_csv(
    path: Path, *, max_rows: int, max_file_bytes: int
) -> tuple[list[dict[str, str]], set[str], str]:
    with _open_checked(path, max_file_bytes=max_file_bytes) as binary:
        stream = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        try:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} has no CSV header")
            rows = []
            for row_number, row in enumerate(reader, 1):
                if row_number > max_rows:
                    raise _ResourceLimitExceeded(
                        f"{path.name} exceeds maximum CSV row count ({max_rows})."
                    )
                rows.append(row)
            columns = set(reader.fieldnames)
        finally:
            stream.detach()
        return rows, columns, _hash_stream(binary)


def _json(path: Path, *, max_file_bytes: int) -> tuple[Any, str]:
    with _open_checked(path, max_file_bytes=max_file_bytes) as binary:
        stream = io.TextIOWrapper(binary, encoding="utf-8")
        try:
            value = json.load(stream)
        finally:
            stream.detach()
        return value, _hash_stream(binary)


def _read_jsonl(
    path: Path, *, max_line_bytes: int, max_records: int, max_file_bytes: int
) -> tuple[list[dict[str, Any]], str]:
    records = []
    with _open_checked(path, max_file_bytes=max_file_bytes) as binary:
        stream = io.TextIOWrapper(binary, encoding="utf-8")
        try:
            for line_no, line in enumerate(stream, 1):
                if len(line.encode("utf-8")) > max_line_bytes:
                    raise _ResourceLimitExceeded(
                        f"{path.name} line {line_no} exceeds maximum JSONL line size "
                        f"({max_line_bytes} bytes)."
                    )
                if not line.strip():
                    continue
                if len(records) >= max_records:
                    raise _ResourceLimitExceeded(
                        f"{path.name} exceeds maximum JSONL record count ({max_records})."
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name} line {line_no} must be an object")
                records.append(value)
        finally:
            stream.detach()
        return records, _hash_stream(binary)


def _positive_int(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value.strip()) is not None


def _aware_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_timezone(value: object) -> bool:
    if not _nonempty(value):
        return False
    try:
        ZoneInfo(value.strip())
    except (ZoneInfoNotFoundError, TypeError, ValueError):
        return False
    return True


def _artifact_path(root: Path, name: str) -> Path:
    """Resolve flat packages and the documented split data/embedding layout."""
    candidates = (root / name, root / "data" / name, root / "embeddings" / name)
    for candidate in candidates:
        if candidate.is_symlink():
            return candidate
        if candidate.is_file():
            return candidate
    return root / name


def _path_security_failures(root: Path, paths: dict[str, Path]) -> list[str]:
    failures = []
    root_resolved = root.resolve()
    for name, path in paths.items():
        try:
            if path.is_symlink():
                failures.append(f"{name} is a symlink.")
                continue
            resolved = path.resolve(strict=True)
            resolved.relative_to(root_resolved)
            if not path.is_file():
                failures.append(f"{name} is not a regular file.")
        except (OSError, ValueError) as exc:
            failures.append(f"{name} is outside the package or cannot be validated: {exc}.")
    return failures


def _hash_stream(stream) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _check_file_sizes(
    paths: dict[str, Path], *, max_file_bytes: int
) -> list[str]:
    failures = []
    for name, path in paths.items():
        try:
            size = path.stat().st_size
        except OSError as exc:
            failures.append(f"{name} size could not be inspected: {exc}.")
            continue
        if size > max_file_bytes:
            failures.append(
                f"{name} exceeds maximum file size ({max_file_bytes} bytes)."
            )
    return failures


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


class _TensorMetadataUnpickler(pickle.Unpickler):
    """Read tensor shape metadata without constructing tensor storage."""

    def __init__(self, file, *, max_elements: int):
        super().__init__(file)
        self.max_elements = max_elements

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
            "_rebuild_parameter",
        }:
            return self._rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return object
        raise _UnsupportedTensorFormat(
            f"Unsupported tensor metadata global: {module}.{name}."
        )

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5:
            raise _UnsupportedTensorFormat("Unsupported tensor storage metadata.")
        storage_size = pid[4]
        if not isinstance(storage_size, int) or storage_size < 0:
            raise _UnsupportedTensorFormat("Invalid tensor storage size metadata.")
        if storage_size > self.max_elements:
            raise _ResourceLimitExceeded(
                "embeddings.pt storage exceeds maximum tensor element count "
                f"({self.max_elements})."
            )
        return storage_size

    @staticmethod
    def _rebuild_tensor(storage, *args):
        size = args[1] if len(args) >= 2 else None
        if not isinstance(size, (tuple, list)) or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in size
        ):
            raise _UnsupportedTensorFormat("Invalid tensor shape metadata.")
        return _TensorMetadata(tuple(size))


def _inspect_tensor_bytes(
    data: bytes, *, expected_rows: int, limits: ArtifactValidationLimits
) -> tuple[int, ...]:
    """Preflight supported torch zip tensors before tensor storage is loaded."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            metadata_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if len(metadata_names) != 1:
                raise _UnsupportedTensorFormat(
                    "embeddings.pt must be a single torch zip tensor package."
                )
            info = archive.getinfo(metadata_names[0])
            if info.file_size > limits.max_file_bytes:
                raise _ResourceLimitExceeded(
                    "embeddings.pt metadata exceeds the maximum file size."
                )
            metadata = archive.read(metadata_names[0])
    except zipfile.BadZipFile as exc:
        raise _UnsupportedTensorFormat(
            "embeddings.pt is not a supported torch zip tensor package."
        ) from exc

    try:
        value = _TensorMetadataUnpickler(
            io.BytesIO(metadata), max_elements=limits.max_tensor_elements
        ).load()
    except (pickle.PickleError, EOFError, IndexError, TypeError) as exc:
        raise _UnsupportedTensorFormat(
            "embeddings.pt tensor metadata could not be safely inspected."
        ) from exc
    if isinstance(value, _TensorMetadata):
        tensor_metadata = value
    elif isinstance(value, dict) and isinstance(value.get("embeddings"), _TensorMetadata):
        tensor_metadata = value["embeddings"]
    else:
        raise _UnsupportedTensorFormat(
            "embeddings.pt must contain a tensor or an embeddings tensor entry."
        )
    if tensor_metadata.shape != (expected_rows, EMBEDDING_DIM):
        raise ValueError(
            f"Tensor metadata shape {list(tensor_metadata.shape)} does not match "
            f"expected [{expected_rows}, {EMBEDDING_DIM}]."
        )
    if tensor_metadata.numel > limits.max_tensor_elements:
        raise _ResourceLimitExceeded(
            "embeddings.pt exceeds maximum tensor element count "
            f"({limits.max_tensor_elements})."
        )
    return tensor_metadata.shape


def validate_nlp_artifact_package(
    package_dir: str | Path,
    *,
    limits: ArtifactValidationLimits | None = None,
) -> dict[str, Any]:
    """Return deterministic PASS/FAIL/BLOCKED check results for a package."""
    limits = limits or DEFAULT_LIMITS
    root = Path(package_dir)
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    manifest_name = "artifact_manifest.json" if (root / "artifact_manifest.json").is_file() else "manifest.json"
    paths = {name: _artifact_path(root, name) for name in REQUIRED_FILES}
    paths[manifest_name] = _artifact_path(root, manifest_name)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        add("required_files", "BLOCKED", "Missing: " + ", ".join(missing))
        return {"status": "BLOCKED", "genuine_data_claim": False, "checks": checks,
                "blockers": [f"Missing required artifact: {name}" for name in missing]}
    add("required_files", "PASS", "All seven required package files are present.")

    path_failures = _path_security_failures(root, paths)
    if path_failures:
        detail = " ".join(path_failures)
        add("artifact_paths", "FAIL", detail)
        return {"status": "FAIL", "genuine_data_claim": False, "checks": checks,
                "blockers": [f"artifact_paths: {detail}"]}

    size_failures = _check_file_sizes(paths, max_file_bytes=limits.max_file_bytes)
    if size_failures:
        detail = " ".join(size_failures)
        add("resource_limits", "FAIL", detail)
        return {"status": "FAIL", "genuine_data_claim": False, "checks": checks,
                "blockers": [f"resource_limits: {detail}"]}

    try:
        metadata, meta_columns, metadata_hash = _read_csv(
            paths["embedding_metadata.csv"], max_rows=limits.max_csv_rows,
            max_file_bytes=limits.max_file_bytes,
        )
        logs, log_columns, logs_hash = _read_csv(
            paths["processed_logs.csv"], max_rows=limits.max_csv_rows,
            max_file_bytes=limits.max_file_bytes,
        )
        severity, severity_columns, severity_hash = _read_csv(
            paths["severity.csv"], max_rows=limits.max_csv_rows,
            max_file_bytes=limits.max_file_bytes,
        )
        report, report_hash = _json(
            paths["verification_report.json"], max_file_bytes=limits.max_file_bytes
        )
        manifest, manifest_hash = _json(
            paths[manifest_name], max_file_bytes=limits.max_file_bytes
        )
        entities, entities_hash = _read_jsonl(
            paths["entities.jsonl"],
            max_line_bytes=limits.max_jsonl_line_bytes,
            max_records=limits.max_jsonl_records,
            max_file_bytes=limits.max_file_bytes,
        )
        with _open_checked(
            paths["embeddings.pt"], max_file_bytes=limits.max_file_bytes
        ) as tensor_stream:
            tensor_bytes = tensor_stream.read()
            tensor_hash = _hash_stream(tensor_stream)
        _inspect_tensor_bytes(
            tensor_bytes, expected_rows=len(metadata), limits=limits
        )
        # weights_only avoids accepting arbitrary pickled Python objects. The
        # tensor bytes were preflighted before this load and are immutable in memory.
        embeddings = torch.load(
            io.BytesIO(tensor_bytes), map_location="cpu", weights_only=True
        )
        if isinstance(embeddings, dict) and "embeddings" in embeddings:
            embeddings = embeddings["embeddings"]
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("embeddings.pt must contain a tensor")
        if embeddings.numel() > limits.max_tensor_elements:
            raise _ResourceLimitExceeded(
                "embeddings.pt exceeds maximum tensor element count "
                f"({limits.max_tensor_elements})."
            )
        if not isinstance(report, dict) or not isinstance(manifest, dict):
            raise ValueError("verification report and artifact manifest must be JSON objects")
        actual_hashes = {
            "embedding_metadata.csv": metadata_hash,
            "processed_logs.csv": logs_hash,
            "severity.csv": severity_hash,
            "entities.jsonl": entities_hash,
            "embeddings.pt": tensor_hash,
            "verification_report.json": report_hash,
            manifest_name: manifest_hash,
        }
    except _ResourceLimitExceeded as exc:
        add("resource_limits", "FAIL", str(exc))
        return {"status": "FAIL", "genuine_data_claim": False, "checks": checks,
                "blockers": [f"resource_limits: {exc}"]}
    except Exception as exc:
        add("package_parse", "FAIL", str(exc))
        return {"status": "FAIL", "genuine_data_claim": False, "checks": checks,
                "blockers": ["One or more package files could not be parsed."]}
    add("resource_limits", "PASS", "All configured artifact resource limits were respected.")
    add("package_parse", "PASS", "CSV, JSON, JSONL, and tensor files parsed.")

    dataset_type = report.get("dataset_type")
    if dataset_type == "synthetic_fixture":
        classification_status = "PASS"
        classification_detail = "Package is explicitly classified as synthetic_fixture; no genuine-data claim is made."
    elif dataset_type in (None, ""):
        classification_status = "BLOCKED"
        classification_detail = "Dataset classification is absent; synthetic and genuine data cannot be distinguished."
    else:
        classification_status = "BLOCKED"
        classification_detail = f"Dataset classification {dataset_type!r} is not trusted evidence of genuine data."
    add("dataset_classification", classification_status, classification_detail)

    synthetic_contract_ok = True
    synthetic_contract_detail = "Synthetic source and namespace fields are consistent."
    if dataset_type == "synthetic_fixture":
        source_splits = [
            row.get("source_split", "") for row in (*metadata, *logs)
        ]
        namespaces = [
            row.get("namespace", "")
            for row in (*metadata, *logs)
            if "namespace" in row
        ]
        synthetic_contract_ok = (
            all(value == "fixture" for value in source_splits)
            and all(value == "synthetic_fixture" for value in namespaces)
        )
        if not synthetic_contract_ok:
            synthetic_contract_detail = (
                "Synthetic artifacts must use source_split='fixture' and, when "
                "present, namespace='synthetic_fixture'."
            )
    add(
        "synthetic_contract",
        "PASS" if synthetic_contract_ok else "FAIL",
        synthetic_contract_detail,
    )

    missing_cols = sorted(set(METADATA_FIELDS) - meta_columns)
    add("metadata_schema", "PASS" if not missing_cols else "FAIL",
        "Required metadata columns present." if not missing_cols else f"Missing columns: {missing_cols}")
    n = len(metadata)
    shape_ok = embeddings.ndim == 2 and embeddings.shape == (n, EMBEDDING_DIM)
    add("embedding_shape", "PASS" if shape_ok else "FAIL",
        f"Tensor shape {list(embeddings.shape)}; metadata rows {n}; expected [N, 768].")
    finite = bool(torch.is_floating_point(embeddings) and torch.isfinite(embeddings).all().item())
    add("embedding_finite", "PASS" if finite else "FAIL", "All embedding values are finite floating point." if finite else "Embedding values must be finite floating point numbers.")

    metadata_values_ok = all(
        _nonempty(row.get(field))
        for row in metadata
        for field in METADATA_FIELDS
        if field != "row_index"
    )
    add("metadata_values", "PASS" if metadata_values_ok else "FAIL",
        "Required metadata values are non-empty." if metadata_values_ok else
        "One or more required metadata values are blank or malformed.")

    row_indexes: list[int] = []
    ids: list[str] = []
    row_errors = []
    for pos, row in enumerate(metadata):
        try:
            idx = int(row.get("row_index", ""))
            if str(idx) != row.get("row_index", "").strip() or idx < 0 or idx >= n:
                raise ValueError
            row_indexes.append(idx)
        except ValueError:
            row_errors.append(f"invalid row_index at metadata row {pos}")
        ids.append(row.get("log_id", "").strip())
    row_ok = not row_errors and len(set(row_indexes)) == n and set(row_indexes) == set(range(n))
    add("row_index_coverage", "PASS" if row_ok else "FAIL",
        "row_index is unique, in bounds, and covers every tensor row." if row_ok else "; ".join(row_errors) or "row_index is duplicate or incomplete.")
    add("duplicate_log_ids", "PASS" if all(ids) and len(ids) == len(set(ids)) else "FAIL",
        "Metadata log_id values are non-empty and unique." if all(ids) and len(ids) == len(set(ids)) else "Metadata contains blank or duplicate log_id values.")

    log_ids = [r.get("log_id", "").strip() for r in logs]
    ordered_ids = [
        ids[index] for index in row_indexes
    ] if row_ok else []
    same_log_ids = (
        len(logs) == n and len(set(log_ids)) == len(log_ids)
        and ordered_ids == log_ids and set(log_ids) == set(ids)
    )
    add("processed_log_alignment", "PASS" if same_log_ids else "FAIL",
        "Processed logs and metadata have a one-to-one log_id mapping." if same_log_ids else "Processed log IDs/count do not match embedding metadata.")
    log_map = {r.get("log_id", "").strip(): r for r in logs}
    fields_consistent = same_log_ids and all(
        all(
            row.get(field, "") == log_map[row.get("log_id", "").strip()].get(field, "")
            for field in ("unit_id", "cycle", "source_split")
        )
        for row in metadata
    )
    add("metadata_log_consistency", "PASS" if fields_consistent else "FAIL",
        "unit_id, cycle, and source_split agree across metadata and processed logs." if fields_consistent else "Metadata and processed-log fields disagree.")

    unit_cycle_ok = (
        all(_positive_int(row.get("unit_id")) and _positive_int(row.get("cycle")) for row in metadata)
        and all(_positive_int(row.get("unit_id")) and _positive_int(row.get("cycle")) for row in logs)
    )
    add("unit_cycle", "PASS" if unit_cycle_ok else "FAIL",
        "Every metadata row supplies positive unit_id and cycle values." if unit_cycle_ok else "Missing or invalid unit_id/cycle; values were not inferred.")

    absent_provenance = sorted(set(PROVENANCE_FIELDS) - (meta_columns | log_columns))
    provenance_ok = not absent_provenance and all(
        all(row.get(field, "").strip() for field in PROVENANCE_FIELDS)
        for row in metadata
    )
    add("source_provenance", "PASS" if provenance_ok else "BLOCKED",
        "Source identity fields are present." if provenance_ok else f"Source provenance unavailable or blank: {absent_provenance or 'one or more metadata rows'}.")

    revisions_ok = all(all(_nonempty(row.get(field)) for field in REVISION_FIELDS) for row in metadata)
    add("model_tokenizer_revisions", "PASS" if revisions_ok else "BLOCKED",
        "Model/tokenizer names and immutable revisions are present." if revisions_ok else "Model or tokenizer revision evidence is missing; names alone do not identify immutable revisions.")

    timestamp_cols = sorted(set(TIMESTAMP_FIELDS) - log_columns)
    timestamp_ok = not timestamp_cols and all(
        _aware_timestamp(row.get(field)) for row in logs for field in TIMESTAMP_FIELDS
    )
    timezone_values = [row.get("timezone", "") for row in logs if "timezone" in log_columns]
    timezone_ok = not timezone_values or all(_valid_timezone(value) for value in timezone_values)
    timestamp_ok = timestamp_ok and timezone_ok
    add("timestamps", "PASS" if timestamp_ok else "BLOCKED",
        "All documented timestamp values are parseable and timezone-aware." if timestamp_ok else f"Timezone-aware timestamp evidence unavailable or invalid; missing columns: {timestamp_cols}.")

    duplicate_records = (
        len({json.dumps(row, sort_keys=True) for row in metadata}) != n
        or len({json.dumps(row, sort_keys=True) for row in logs}) != len(logs)
        or len({json.dumps(row, sort_keys=True) for row in severity}) != len(severity)
        or len({json.dumps(item, sort_keys=True) for item in entities}) != len(entities)
    )
    add("duplicate_records", "FAIL" if duplicate_records else "PASS",
        "No duplicate metadata or processed-log rows." if not duplicate_records else "Duplicate records detected.")
    entity_ids = [str(item.get("log_id", "")).strip() for item in entities]
    severity_ids = [r.get("log_id", "").strip() for r in severity]
    related_ok = (
        all(x in set(ids) for x in entity_ids if x)
        and all(x in set(ids) for x in severity_ids if x)
        and all(entity_ids.count(x) >= 1 for x in entity_ids if x)
    )
    add("related_artifact_ids", "PASS" if related_ok else "FAIL",
        "Entity and severity references resolve to metadata log IDs." if related_ok else "Entity/severity records contain unknown log IDs.")
    add("severity_schema", "PASS" if "log_id" in severity_columns else "BLOCKED",
        "Severity rows have a log_id key." if "log_id" in severity_columns else "Severity schema lacks log_id; record linkage cannot be checked.")
    add("entity_schema", "PASS" if "log_id" in {key for item in entities for key in item} or not entities else "BLOCKED",
        "Entity records expose log_id linkage." if not entities or "log_id" in {key for item in entities for key in item} else "Entity records lack log_id linkage.")

    def related_fields_match(rows: list[dict[str, str]]) -> bool:
        return all(
            row.get("log_id", "").strip() in log_map
            and all(row.get(field, "") == log_map[row.get("log_id", "").strip()].get(field, "")
                    for field in ("unit_id", "cycle", "source_split") if field in row)
            for row in rows
        )

    add("related_record_consistency", "PASS" if related_fields_match(severity) and related_fields_match(entities) else "FAIL",
        "Entity and severity unit/cycle/split values agree with processed logs." if related_fields_match(severity) and related_fields_match(entities) else
        "Entity or severity unit/cycle/split values disagree with processed logs.")

    # Hash entries may use a {files: {name: sha256}} or direct filename mapping.
    hash_map = manifest.get("files", manifest.get("artifact_hashes", {}))
    if not isinstance(hash_map, dict):
        hash_map = {}
    hash_failures = []
    hash_checked = 0
    for name, entry in hash_map.items():
        name = Path(name).name
        if name not in paths:
            continue
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        if expected is None:
            continue
        hash_checked += 1
        if (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)
            or actual_hashes[name].lower()
            != expected.lower()
        ):
            hash_failures.append(name)
    add("manifest_hashes", "FAIL" if hash_failures else ("PASS" if hash_checked else "BLOCKED"),
        f"Hash mismatch/invalid digest for: {hash_failures}." if hash_failures else (f"Validated {hash_checked} manifest-provided SHA-256 hashes." if hash_checked else "Manifest provides no verifiable hashes for package files."))

    # Upstream report values are consistency evidence, not cryptographic proof.
    report_consistent = (
        report.get("embedding_shape") == list(embeddings.shape)
        and report.get("number_of_logs") == n
        and report.get("embedding_finite") is True
    )
    add("upstream_report_consistency", "PASS" if report_consistent else "FAIL",
        "Report row count and embedding shape agree with package." if report_consistent else "Upstream report disagrees with package contents.")
    add("authenticity", "BLOCKED", "An unsigned manifest/report cannot establish cryptographic authenticity.")
    blockers = [f"{c['name']}: {c['detail']}" for c in checks if c["status"] == "BLOCKED"]
    failures = any(c["status"] == "FAIL" for c in checks)
    # Synthetic packages may be structurally valid while authenticity and
    # genuine-only evidence remain blocked. That distinction is explicit in
    # the report and does not establish genuine-data authenticity.
    status = "FAIL" if failures else (
        "PASS" if dataset_type == "synthetic_fixture" else
        "BLOCKED" if blockers else "PASS"
    )
    return {"status": status, "genuine_data_claim": False, "checks": checks, "blockers": blockers}


def load_validated_nlp_artifact_package(
    package_dir: str | Path,
    *,
    limits: ArtifactValidationLimits | None = None,
) -> dict[str, Any]:
    """Validate, then load package inputs for an enforced integration boundary."""
    report = validate_nlp_artifact_package(package_dir, limits=limits)
    if report.get("status") != "PASS":
        raise ValueError(
            "NLP artifact validation did not PASS; fusion ingestion is blocked: "
            + "; ".join(report.get("blockers", []))
        )

    limits = limits or DEFAULT_LIMITS
    root = Path(package_dir)
    manifest_name = (
        "artifact_manifest.json"
        if (root / "artifact_manifest.json").is_file()
        else "manifest.json"
    )
    paths = {name: _artifact_path(root, name) for name in REQUIRED_FILES}
    paths[manifest_name] = _artifact_path(root, manifest_name)
    source_report, _ = _json(
        paths["verification_report.json"],
        max_file_bytes=limits.max_file_bytes,
    )
    metadata, _, _ = _read_csv(
        paths["embedding_metadata.csv"],
        max_rows=limits.max_csv_rows,
        max_file_bytes=limits.max_file_bytes,
    )
    processed_logs, _, _ = _read_csv(
        paths["processed_logs.csv"],
        max_rows=limits.max_csv_rows,
        max_file_bytes=limits.max_file_bytes,
    )
    severity, _, _ = _read_csv(
        paths["severity.csv"],
        max_rows=limits.max_csv_rows,
        max_file_bytes=limits.max_file_bytes,
    )
    entities, _ = _read_jsonl(
        paths["entities.jsonl"],
        max_line_bytes=limits.max_jsonl_line_bytes,
        max_records=limits.max_jsonl_records,
        max_file_bytes=limits.max_file_bytes,
    )
    with _open_checked(
        paths["embeddings.pt"], max_file_bytes=limits.max_file_bytes
    ) as tensor_stream:
        tensor_bytes = tensor_stream.read()
    _inspect_tensor_bytes(
        tensor_bytes, expected_rows=len(metadata), limits=limits
    )
    embeddings = torch.load(
        io.BytesIO(tensor_bytes), map_location="cpu", weights_only=True
    )
    if isinstance(embeddings, dict) and "embeddings" in embeddings:
        embeddings = embeddings["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("embeddings.pt must contain a tensor")
    return {
        "metadata": metadata,
        "processed_logs": processed_logs,
        "entities": entities,
        "severity": severity,
        "embeddings": embeddings,
        "verification_report": source_report,
        "validation_report": report,
    }
