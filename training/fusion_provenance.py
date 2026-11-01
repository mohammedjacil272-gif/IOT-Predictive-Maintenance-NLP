"""Fail-closed provenance and point-in-time checks for fusion inputs."""

from __future__ import annotations

import math
import re
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


def validate_genuine_approval(approval: Mapping | None) -> None:
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
