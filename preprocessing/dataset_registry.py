"""Small, reusable contracts for namespaced external-dataset intake.

This module records metadata and file hashes only. It never downloads data,
infers relationships, or treats a dataset as genuine based on its label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


class DatasetContractError(ValueError):
    """Raised when dataset metadata or a supplied record violates the contract."""


_NAMESPACE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_CLASSIFICATIONS = {"benchmark", "genuine", "synthetic_fixture", "derived"}


@dataclass(frozen=True)
class DatasetDescriptor:
    namespace: str
    version: str
    classification: str
    source: str
    license: str

    def __post_init__(self) -> None:
        if not _NAMESPACE.fullmatch(self.namespace):
            raise DatasetContractError("namespace must be 3-64 lowercase characters or underscores")
        if not self.version.strip():
            raise DatasetContractError("version must be non-empty")
        if self.classification not in _CLASSIFICATIONS:
            raise DatasetContractError(
                f"classification must be one of {sorted(_CLASSIFICATIONS)}"
            )
        if not self.source.strip() or not self.license.strip():
            raise DatasetContractError("source and license are required")

    def to_dict(self) -> dict[str, str]:
        return {
            "namespace": self.namespace,
            "version": self.version,
            "classification": self.classification,
            "source": self.source,
            "license": self.license,
        }


def sha256_file(path: str | Path) -> str:
    """Hash one regular, non-symlink file without modifying it."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise DatasetContractError(f"dataset file must be a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_files(root: str | Path) -> list[dict[str, Any]]:
    """Return a deterministic hash inventory for regular files below *root*."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise DatasetContractError(f"dataset root is not a directory: {base}")
    inventory: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise DatasetContractError(f"symlinks are not allowed in raw intake: {path}")
        if path.is_file():
            inventory.append({
                "relative_path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return inventory


def validate_record_namespace(record: Mapping[str, Any], descriptor: DatasetDescriptor) -> None:
    """Require every normalized record to carry the descriptor namespace."""
    if record.get("dataset_namespace") != descriptor.namespace:
        raise DatasetContractError(
            f"record dataset_namespace must equal {descriptor.namespace!r}"
        )


def validate_unique_keys(records: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> None:
    """Reject missing or duplicate record keys without silently dropping rows."""
    seen: set[tuple[Any, ...]] = set()
    for row_number, record in enumerate(records, 1):
        try:
            key = tuple(record[key] for key in keys)
        except KeyError as exc:
            raise DatasetContractError(
                f"record {row_number} is missing key field {exc.args[0]!r}"
            ) from exc
        if any(value is None or (isinstance(value, str) and not value.strip()) for value in key):
            raise DatasetContractError(f"record {row_number} has a blank identity key")
        if key in seen:
            raise DatasetContractError(f"duplicate record key at row {row_number}: {key!r}")
        seen.add(key)
