"""Deterministic, equipment-level fusion split manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence


def canonical_manifest(
    assignments: Mapping[str, Sequence[int]],
    *,
    namespace: str,
    expected_equipment_ids: Sequence[int] | None = None,
    test_approval: Mapping | None = None,
) -> dict:
    """Validate and canonicalize equipment assignments, then attach identity/hash."""
    allowed = {"train", "validation", "test"}
    if namespace not in {"genuine", "synthetic_fixture"}:
        raise ValueError("Split manifest namespace must be genuine or synthetic_fixture.")
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("Split assignments must be a non-empty mapping.")
    normalized: dict[str, list[int]] = {}
    owners: dict[int, str] = {}
    for split, values in assignments.items():
        if split not in allowed:
            raise ValueError(f"Unsupported fusion split: {split!r}.")
        units = []
        for raw in values:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ValueError(f"Invalid equipment ID in {split}: {raw!r}.")
            if raw in owners:
                raise ValueError(f"Equipment leakage for unit {raw}: {owners[raw]} and {split}.")
            owners[raw] = split
            units.append(raw)
        if len(units) != len(set(units)):
            raise ValueError(f"Duplicate equipment ID within {split}.")
        normalized[split] = sorted(units)
    if not normalized.get("train") or not normalized.get("validation"):
        raise ValueError("Fusion split manifest requires non-empty train and validation sets.")
    if expected_equipment_ids is not None:
        expected = set()
        for raw in expected_equipment_ids:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ValueError(f"Invalid expected equipment ID: {raw!r}.")
            expected.add(raw)
        if len(expected) != len(expected_equipment_ids):
            raise ValueError("Duplicate expected equipment IDs.")
        assigned = set(owners)
        missing = sorted(expected - assigned)
        unexpected = sorted(assigned - expected)
        if missing or unexpected:
            raise ValueError(
                f"Split assignment coverage mismatch; missing={missing}, unexpected={unexpected}."
            )

    approval = None
    if normalized.get("test"):
        approval = test_approval
        if not (
            isinstance(approval, Mapping)
            and approval.get("status") == "APPROVED"
            and isinstance(approval.get("approved_by"), str)
            and approval["approved_by"].strip()
            and isinstance(approval.get("evidence_refs"), list)
            and approval["evidence_refs"]
            and all(isinstance(ref, str) and ref.strip() for ref in approval["evidence_refs"])
        ):
            raise ValueError("A final test composition cannot be claimed approved without an explicit evidenced approval.")

    core = {
        "schema": "fusion-equipment-split-v1",
        "namespace": namespace,
        "assignments": {key: normalized[key] for key in sorted(normalized)},
        "final_test_composition_approval": dict(approval) if approval else {"status": "PENDING"},
    }
    payload = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "identity": hashlib.sha256(payload).hexdigest(), "sha256": hashlib.sha256(payload).hexdigest()}
