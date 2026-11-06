"""Integrity receipts for multimodal fusion samples."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping

import torch


RECEIPT_KEY = "_integrity_receipt"
RECEIPT_VERSION = 1

TENSOR_FIELDS = (
    "unit_id",
    "cycle",
    "sensor_embeddings",
    "text_embeddings",
    "rul",
    "has_log",
)

METADATA_FIELDS = (
    "namespace",
    "split",
    "log_ids_by_window",
    "entities_by_window",
    "severity_by_log_by_window",
    "timestamps_by_log_by_window",
    "raw_timestamps_by_log_by_window",
    "source_ids_by_window",
    "sensor_provenance_by_window",
    "window_severity",
    "excluded_logs",
)


def _canonical_json(value: Any) -> bytes:
    """Serialize metadata deterministically."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Fusion integrity metadata is not canonicalizable.") from exc


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Unsupported integrity metadata type: {type(value).__name__}")


def compute_samples_digest(samples: Mapping[str, Any]) -> str:
    """Hash all training tensors and relevant alignment/provenance metadata."""
    digest = hashlib.sha256()
    digest.update(f"fusion-samples-integrity-v{RECEIPT_VERSION}".encode("utf-8"))

    for field in TENSOR_FIELDS:
        if field not in samples:
            raise ValueError(f"Fusion samples missing integrity tensor: {field}")

        tensor = samples[field]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Fusion integrity field {field} is not a tensor.")

        canonical_tensor = tensor.detach().cpu().contiguous()
        digest.update(field.encode("utf-8"))
        digest.update(str(canonical_tensor.dtype).encode("utf-8"))
        digest.update(_canonical_json(list(canonical_tensor.shape)))
        digest.update(canonical_tensor.numpy().tobytes())

    metadata = {}
    for field in METADATA_FIELDS:
        if field not in samples:
            raise ValueError(f"Fusion samples missing integrity metadata: {field}")
        metadata[field] = samples[field]

    digest.update(_canonical_json(metadata))
    return digest.hexdigest()


def issue_integrity_receipt(samples: dict[str, Any]) -> dict[str, Any]:
    """Attach a receipt after sample construction."""
    if RECEIPT_KEY in samples:
        raise ValueError("Fusion samples already contain an integrity receipt.")

    receipt = {
        "version": RECEIPT_VERSION,
        "algorithm": "sha256",
        "digest": compute_samples_digest(samples),
    }
    samples[RECEIPT_KEY] = receipt
    return receipt


def verify_integrity_receipt(samples: Mapping[str, Any]) -> None:
    """Fail closed if the receipt is absent, malformed, or samples changed."""
    receipt = samples.get(RECEIPT_KEY)

    if not isinstance(receipt, Mapping):
        raise ValueError("Fusion sample integrity receipt is missing.")

    if receipt.get("version") != RECEIPT_VERSION:
        raise ValueError("Unsupported fusion integrity receipt version.")

    if receipt.get("algorithm") != "sha256":
        raise ValueError("Unsupported fusion integrity receipt algorithm.")

    expected = receipt.get("digest")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Malformed fusion integrity receipt digest.")

    actual = compute_samples_digest(samples)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("Fusion sample integrity verification failed.")
