"""Fail-closed provenance and point-in-time checks for fusion inputs."""

from __future__ import annotations

import math
import re
from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


TIMESTAMP_FIELDS = (
    "event_time",
    "record_creation_time",
    "ingestion_time",
    "availability_time",
)

GENUINE_APPROVALS = (
    "dataset_permission",
    "equipment_id_mapping",
    "cycle_semantics",
    "rul_target_definition",
    "timestamp_semantics",
    "prediction_cutoff_and_horizon",
    "split_and_evaluation_protocol",
    "missing_multiple_log_policy",
    "has_log_feature_policy",
    "embedding_provenance",
)
TRAINING_AUTHORIZATION = "fusion_training_authorization"
FINAL_EVALUATION_AUTHORIZATION = "final_held_out_evaluation_authorization"
APPROVED_MANIFEST = "approved_manifest"


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Approved manifest requires non-empty {field}.")
    return value.strip()


def validate_approved_manifest(approval: Mapping) -> Mapping:
    """Validate the immutable lineage manifest attached to a genuine approval."""
    manifest = approval.get(APPROVED_MANIFEST)
    if not isinstance(manifest, Mapping):
        raise ValueError("Genuine-data use is blocked without an approved artifact manifest.")
    for field in ("manifest_id", "equipment_mapping_id"):
        _required_text(manifest.get(field), field)
    datasets = manifest.get("datasets")
    if datasets is not None:
        if not isinstance(datasets, Mapping) or not {"sensor", "text"}.issubset(datasets):
            raise ValueError("Approved manifest datasets must include sensor and text identities.")
        for modality, dataset in datasets.items():
            if not isinstance(dataset, Mapping):
                raise ValueError(f"Malformed approved dataset entry: {modality!r}.")
            for field in ("source_dataset", "source_dataset_version", "preprocessing_version"):
                _required_text(dataset.get(field), f"{modality}.{field}")
    else:
        for field in ("source_dataset", "source_dataset_version", "preprocessing_version"):
            _required_text(manifest.get(field), field)
    scopes = manifest.get("record_scope")
    if not isinstance(scopes, Mapping) or not {"sensor", "text"}.issubset(scopes):
        raise ValueError("Approved manifest must define sensor and text record scopes.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("Approved artifact manifest must enumerate artifacts.")
    for artifact_key, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Malformed approved artifact entry: {artifact_key!r}.")
        for field in ("artifact_id", "artifact_sha256", "artifact_path", "modality", "embedding_model", "embedding_revision"):
            _required_text(artifact.get(field), f"artifact.{field}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", artifact["artifact_sha256"]):
            raise ValueError(f"Approved artifact {artifact_key!r} has an invalid SHA-256 digest.")
        if artifact["modality"] not in {"sensor", "text"}:
            raise ValueError(f"Approved artifact {artifact_key!r} has an invalid modality.")
        binding = artifact.get("embedding_sha256") or artifact.get("embedding_sha256_by_source_record_id")
        if isinstance(binding, str):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", binding):
                raise ValueError(f"Approved artifact {artifact_key!r} has an invalid embedding digest.")
        elif isinstance(binding, Mapping):
            if not binding or any(
                not isinstance(key, str) or not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", value)
                for key, value in binding.items()
            ):
                raise ValueError(f"Approved artifact {artifact_key!r} has invalid record embedding digests.")
        else:
            raise ValueError(f"Approved artifact {artifact_key!r} lacks an embedding digest binding.")
    decision = approval.get("approvals", {}).get("embedding_provenance") if isinstance(approval.get("approvals"), Mapping) else None
    if not isinstance(decision, Mapping) or decision.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("Embedding-provenance approval is not bound to the approved manifest.")
    return manifest


def validate_record_against_manifest(record: Mapping, *, approval: Mapping, modality: str) -> Mapping:
    """Verify record lineage, artifact bytes, and approved modality identity."""
    manifest = validate_approved_manifest(approval)
    if modality not in {"sensor", "text"}:
        raise ValueError(f"Unsupported provenance modality: {modality!r}.")
    dataset = manifest.get("datasets", {}).get(modality) if isinstance(manifest.get("datasets"), Mapping) else manifest
    common = {
        "source_dataset": dataset["source_dataset"],
        "source_dataset_version": dataset["source_dataset_version"],
        "equipment_mapping_id": manifest["equipment_mapping_id"],
        "preprocessing_version": dataset["preprocessing_version"],
    }
    for field, expected in common.items():
        if record.get(field) != expected:
            raise ValueError(f"Record {field} does not match the approved manifest.")
    artifact_id = _required_text(record.get("artifact_id"), "record.artifact_id")
    source_record_id = _required_text(record.get("source_record_id"), "record.source_record_id")
    scope = manifest["record_scope"].get(modality)
    if not isinstance(scope, Mapping):
        raise ValueError(f"Approved manifest has no record scope for {modality}.")
    allowed_ids = scope.get("allowed_source_record_ids")
    if allowed_ids is not None:
        if not isinstance(allowed_ids, (list, tuple, set)) or source_record_id not in allowed_ids:
            raise ValueError("Record source_record_id is outside the approved manifest scope.")
    elif scope.get("scope_type") != "dataset":
        raise ValueError("Approved manifest record scope is incomplete.")
    artifact = manifest["artifacts"].get(artifact_id)
    if not isinstance(artifact, Mapping) or artifact.get("modality") != modality:
        raise ValueError("Record artifact is not included for this modality in the approved manifest.")
    for field in ("artifact_id", "artifact_sha256", "artifact_path", "embedding_model", "embedding_revision"):
        if record.get(field) != artifact.get(field):
            raise ValueError(f"Record {field} does not match the approved artifact manifest.")
    path = Path(artifact["artifact_path"])
    if not path.is_file():
        raise ValueError(f"Approved artifact is missing or unreadable: {path}")
    actual = sha256(path.read_bytes()).hexdigest()
    if actual.lower() != artifact["artifact_sha256"].lower():
        raise ValueError("Approved artifact bytes do not match the manifest digest.")
    return manifest


def validate_reproducibility_contract(config: Mapping) -> None:
    """Require approved source versions and explicitly recordable lineage."""
    required_text = (
        "preprocessing_version",
        "nlp_model_name",
        "nlp_model_revision",
        "tokenizer_name",
        "tokenizer_revision",
    )
    missing = [
        field for field in required_text
        if not isinstance(config.get(field), str) or not config[field].strip()
    ]
    if missing:
        raise ValueError(f"Reproducibility contract is missing: {missing}.")
    for field in ("nlp_model_sha256", "tokenizer_sha256"):
        value = config.get(field)
        if value is not None and (
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)
        ):
            raise ValueError(f"{field} must be null or a SHA-256 hex digest.")
    checkpoint = config.get("checkpoint_lineage")
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("source_checkpoint_id"), str
    ) or not checkpoint["source_checkpoint_id"].strip():
        raise ValueError("Reproducibility contract requires source checkpoint lineage.")
    checkpoint_hash = checkpoint.get("source_checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", checkpoint_hash
    ):
        raise ValueError("Owner-approved source_checkpoint_sha256 is required.")


def validate_operational_horizon(value: object) -> dict:
    """Validate an explicitly selected horizon; never choose a default."""
    if not isinstance(value, Mapping):
        raise ValueError("An explicitly configured operational_horizon is required.")
    amount, unit = value.get("value"), value.get("unit")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError("operational_horizon.value must be a finite positive number.")
    if not math.isfinite(float(amount)) or amount <= 0:
        raise ValueError("operational_horizon.value must be a finite positive number.")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("operational_horizon.unit must be explicitly configured.")
    return {"value": float(amount), "unit": unit.strip()}


def validate_verification_report(
    report: Mapping,
    *,
    git_commit: str,
    python_version: str,
    python_executable: str,
) -> None:
    """Require reproducible, passing pre-run test and verifier evidence."""
    if report.get("git_commit") != git_commit:
        raise ValueError("Verification report commit does not match the training source commit.")
    if report.get("python_version") != python_version:
        raise ValueError("Verification report Python version does not match the run environment.")
    if report.get("python_executable") != python_executable:
        raise ValueError("Verification report interpreter does not match the run interpreter.")
    command = report.get("test_command")
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise ValueError("Verification report must include the exact test command.")
    if report.get("test_status") != "PASS" or not isinstance(
        report.get("test_passed"), int
    ) or report["test_passed"] < 1 or report.get("test_failed") != 0:
        raise ValueError("Verification report must record passing test results.")
    checks = report.get("checks")
    required_checks = {
        "fusion_provenance",
        "nlp_fd001_compatibility",
        "baseline_protection",
    }
    if not isinstance(checks, Mapping) or any(
        checks.get(name) != "PASS" for name in required_checks
    ):
        raise ValueError("Verification report is missing a required passing verification.")


def parse_aware_timestamp(value: object, *, field: str, timezone_name: str) -> datetime:
    """Parse an ISO timestamp, requiring an offset and validating its zone."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Malformed {field}: {value!r}") from exc
    else:
        raise ValueError(f"Missing or malformed {field}.")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timezone-naive {field} is not permitted.")

    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise ValueError("Approved timezone must be a non-empty IANA timezone name.")
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid approved timezone: {timezone_name!r}") from exc
    return parsed.astimezone(timezone.utc)


def validate_point_in_time(
    record: Mapping,
    *,
    cutoff: datetime | str,
    policy: Mapping,
) -> dict[str, datetime | bool]:
    """Validate and preserve distinct timestamps against an explicit policy.

    The policy must come from an approved contract; this helper never selects
    a timezone, cutoff, inclusivity, or late-arrival rule on its own.
    """
    required = {
        "approved",
        "timezone",
        "timestamp_mode",
        "cutoff_inclusive",
        "late_arrival_policy",
    }
    missing = required - policy.keys()
    if missing:
        raise ValueError(f"Point-in-time policy is missing fields: {sorted(missing)}")
    if policy["approved"] is not True:
        raise ValueError("Point-in-time policy is not approved.")
    if policy["timestamp_mode"] != "offset_aware":
        raise ValueError(
            "Approved timestamp_mode must be 'offset_aware'; timezone-naive input is rejected."
        )
    if not isinstance(policy["cutoff_inclusive"], bool):
        raise ValueError("cutoff_inclusive must be an approved boolean value.")
    if policy["late_arrival_policy"] not in {"exclude", "reject"}:
        raise ValueError("Approved late_arrival_policy must be 'exclude' or 'reject'.")
    zone = policy["timezone"]

    parsed = {
        field: parse_aware_timestamp(record.get(field), field=field, timezone_name=zone)
        for field in TIMESTAMP_FIELDS
    }
    ordered_fields = (
        "event_time",
        "record_creation_time",
        "ingestion_time",
        "availability_time",
    )
    for earlier, later in zip(ordered_fields, ordered_fields[1:]):
        if parsed[earlier] > parsed[later]:
            raise ValueError(
                f"Timestamp ordering violation: {earlier} must not be after {later}."
            )
    parsed_cutoff = parse_aware_timestamp(cutoff, field="cutoff", timezone_name=zone)
    is_available = (
        parsed["availability_time"] <= parsed_cutoff
        if policy["cutoff_inclusive"]
        else parsed["availability_time"] < parsed_cutoff
    )
    if not is_available and policy["late_arrival_policy"] == "reject":
        raise ValueError("Log availability is after prediction cutoff; record rejected.")
    parsed["eligible"] = is_available
    return parsed


def validate_genuine_approval(approval: Mapping | None) -> Mapping:
    """Require a recorded approval and evidence reference for every gate."""
    if not isinstance(approval, Mapping):
        raise ValueError("Genuine-data use is blocked: no approval register supplied.")
    register_id = approval.get("approval_register_id")
    if not isinstance(register_id, str) or not register_id.strip():
        raise ValueError("Genuine-data use is blocked without an approval_register_id.")
    decisions = approval.get("approvals")
    if not isinstance(decisions, Mapping):
        raise ValueError("Genuine-data use is blocked: approval decisions are missing.")
    missing = []
    malformed = []
    for name in GENUINE_APPROVALS:
        item = decisions.get(name)
        if not isinstance(item, Mapping) or item.get("status") != "APPROVED":
            missing.append(name)
        elif not _valid_approval_record(item):
            malformed.append(name)
    if missing:
        raise ValueError(
            "Genuine-data use is blocked; approvals are absent or not affirmative: "
            + ", ".join(missing)
        )
    if malformed:
        raise ValueError(
            "Genuine-data use is blocked; approval records need approver and "
            "evidence_refs: " + ", ".join(malformed)
        )
    return validate_approved_manifest(approval)


def _valid_approval_record(item: object) -> bool:
    return bool(
        isinstance(item, Mapping)
        and item.get("status") == "APPROVED"
        and isinstance(item.get("approved_by"), str)
        and item["approved_by"].strip()
        and isinstance(item.get("evidence_refs"), list)
        and item["evidence_refs"]
        and all(isinstance(ref, str) and ref.strip() for ref in item["evidence_refs"])
    )


def validate_named_approval(approval: Mapping | None, decision_name: str) -> None:
    """Require one separately recorded approval decision."""
    if not isinstance(approval, Mapping) or not isinstance(
        approval.get("approval_register_id"), str
    ) or not approval["approval_register_id"].strip():
        raise ValueError("Approval register ID is required.")
    decisions = approval.get("approvals")
    record = decisions.get(decision_name) if isinstance(decisions, Mapping) else None
    if not _valid_approval_record(record):
        raise ValueError(
            f"Required approval is missing, unapproved, or lacks evidence: {decision_name}."
        )
