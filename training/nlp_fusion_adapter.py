"""Translate Rafaz NLP embedding metadata into fusion text records.

This adapter performs only schema mappings established by the NLP contract:
``row_index`` selects a vector from its own embedding tensor and
``source_split`` becomes the fusion ``split`` field. It does not infer genuine
source identity, equipment mapping, cycle semantics, timestamp semantics, or
approval state. It accepts synthetic fixtures only; genuine artifact ingestion
is blocked until a separately approved artifact contract exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import torch


EMBEDDING_DIM = 768


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ValueError(f"{field} must be a positive integer.")
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return parsed


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ValueError(f"{field} must be a non-negative integer.")
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return parsed


def adapt_nlp_embedding_metadata(
    metadata_rows: Sequence[Mapping[str, Any]],
    embeddings: torch.Tensor,
    *,
    processed_logs: Sequence[Mapping[str, Any]],
    verification_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build fusion-shaped text records from NLP metadata and its tensor.

    This boundary intentionally accepts synthetic fixtures only. The output
    namespace is fixed to ``synthetic_fixture`` and every row must use the
    literal source split ``fixture``; callers cannot choose either value.
    ``row_index`` must be sequential in metadata order, must cover the tensor,
    and its log IDs must match the processed-log sequence that was embedded.
    No sensor rows are aligned by metadata or tensor position.

    The unsigned verification report provides consistency validation only; it
    does not establish cryptographic authenticity of the report, tensor, or
    source data. Genuine-data compatibility is not claimed: reports classified
    as genuine are rejected, and genuine inputs require a separately approved
    artifact contract that this adapter does not implement.
    """
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("NLP embeddings must be a PyTorch tensor.")
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"NLP embeddings must have shape [rows, {EMBEDDING_DIM}]."
        )
    if not torch.is_floating_point(embeddings):
        raise ValueError("NLP embeddings must use a floating-point dtype.")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("NLP embeddings contain non-finite values.")
    if not isinstance(metadata_rows, Sequence) or isinstance(
        metadata_rows, (str, bytes)
    ):
        raise ValueError("NLP embedding metadata must be a sequence of mappings.")
    if len(metadata_rows) != embeddings.shape[0]:
        raise ValueError("NLP metadata row count does not match embedding count.")
    if not metadata_rows:
        raise ValueError("NLP embedding metadata is empty.")
    if not isinstance(processed_logs, Sequence) or isinstance(
        processed_logs, (str, bytes)
    ):
        raise ValueError("Processed NLP logs must be a sequence of mappings.")
    if len(processed_logs) != len(metadata_rows):
        raise ValueError("Processed-log row count does not match embedding metadata.")
    if not isinstance(verification_report, Mapping):
        raise ValueError("An upstream NLP verification report is required.")
    _validate_fixture_report(verification_report, embeddings, len(metadata_rows))

    processed_ids = []
    for row_number, row in enumerate(processed_logs):
        if not isinstance(row, Mapping):
            raise ValueError(f"Processed-log row {row_number} must be a mapping.")
        log_id = row.get("log_id")
        if not isinstance(log_id, str) or not log_id.strip():
            raise ValueError(f"Processed-log row {row_number} has an invalid log_id.")
        processed_ids.append(log_id.strip())
    if len(processed_ids) != len(set(processed_ids)):
        raise ValueError("Processed logs contain duplicate log_id values.")

    records: list[dict[str, Any] | None] = [None] * len(metadata_rows)
    seen_log_ids: set[str] = set()

    for row_number, source in enumerate(metadata_rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"NLP metadata row {row_number} must be a mapping.")
        required = {
            "row_index", "log_id", "unit_id", "cycle", "source_split",
            "model_name", "max_length", "pooling", "embedding_dim",
        }
        missing = required - source.keys()
        if missing:
            raise ValueError(
                f"NLP metadata row {row_number} is missing fields: {sorted(missing)}"
            )

        index = _nonnegative_integer(source["row_index"], "row_index")
        if index >= len(metadata_rows):
            raise ValueError(f"NLP row_index is out of bounds: {index}.")
        if records[index] is not None:
            raise ValueError(f"Duplicate NLP row_index: {index}.")
        if index != row_number:
            raise ValueError(
                "NLP metadata row order must match sequential row_index values."
            )

        log_id = source["log_id"]
        if not isinstance(log_id, str) or not log_id.strip():
            raise ValueError(f"NLP metadata row {row_number} has an invalid log_id.")
        log_id = log_id.strip()
        if log_id in seen_log_ids:
            raise ValueError(f"Duplicate NLP log_id: {log_id}.")
        seen_log_ids.add(log_id)
        if log_id != processed_ids[index]:
            raise ValueError(
                f"NLP row_index {index} maps to log_id {log_id!r}, but processed-log "
                f"row {index} has {processed_ids[index]!r}."
            )

        unit_id = _positive_integer(source["unit_id"], "unit_id")
        cycle = _positive_integer(source["cycle"], "cycle")
        source_split = source["source_split"]
        if not isinstance(source_split, str) or not source_split.strip():
            raise ValueError(f"NLP metadata row {row_number} has invalid source_split.")
        source_split = source_split.strip()
        if source_split != source["source_split"]:
            raise ValueError(f"NLP source_split contains surrounding whitespace at row {row_number}.")
        if source_split != "fixture":
            raise ValueError(
                f"Synthetic NLP source_split must be 'fixture' at row {row_number}."
            )
        if "split" in source and source["split"] != source_split:
            raise ValueError(f"NLP split and source_split disagree at row {row_number}.")

        if _positive_integer(source["embedding_dim"], "embedding_dim") != EMBEDDING_DIM:
            raise ValueError(
                f"NLP metadata embedding_dim must be {EMBEDDING_DIM}."
            )
        for field in ("model_name", "pooling"):
            if not isinstance(source[field], str) or not source[field].strip():
                raise ValueError(f"NLP metadata {field} must be a non-empty string.")
        max_length = _positive_integer(source["max_length"], "max_length")
        config = {
            "model_name": source["model_name"].strip(),
            "max_length": max_length,
            "pooling": source["pooling"].strip(),
            "embedding_dim": EMBEDDING_DIM,
        }
        for optional_field in (
            "tokenizer_name", "tokenizer_revision", "model_revision",
            "embedding_model", "embedding_revision",
        ):
            if optional_field in source:
                value = source[optional_field]
                config[optional_field] = (
                    value.strip() if isinstance(value, str) and value.strip()
                    else None
                )
        if "namespace" in source and source["namespace"] != "synthetic_fixture":
            raise ValueError(f"NLP namespace conflicts at metadata row {row_number}.")
        if config.get("embedding_model") and config["embedding_model"] != config["model_name"]:
            raise ValueError(
                f"NLP embedding_model and model_name disagree at row {row_number}."
            )
        if row_number == 0:
            artifact_config = config
        else:
            if config.keys() != artifact_config.keys():
                raise ValueError(
                    "NLP embedding configuration fields differ between rows."
                )
            if config != artifact_config:
                raise ValueError(
                    "NLP embedding configuration is inconsistent across rows."
                )

        record = dict(source)
        record.update({
            "log_id": log_id,
            "unit_id": unit_id,
            "cycle": cycle,
            "split": source_split,
            "namespace": "synthetic_fixture",
            "text_embedding": embeddings[index].detach().to(
                device="cpu", dtype=torch.float32
            ),
        })
        # A model name identifies the family, not an immutable revision.
        # Preserve it under the loader's model-name field without inventing a
        # revision; genuine provenance validation still requires that value.
        record.setdefault("embedding_model", source["model_name"].strip())
        records[index] = record

    if any(record is None for record in records):
        raise ValueError("NLP row_index values do not cover every embedding row.")
    _validate_report_configuration(verification_report, artifact_config)
    _validate_processed_alignment(metadata_rows, processed_logs)
    return [record for record in records if record is not None]


def _validate_fixture_report(
    report: Mapping[str, Any], embeddings: torch.Tensor, row_count: int
) -> None:
    """Check report consistency; this unsigned report is not authenticity evidence."""
    if report.get("dataset_type") != "synthetic_fixture":
        raise ValueError(
            "NLP fusion adapter accepts only artifacts classified as synthetic_fixture."
        )
    reported_count = report.get("number_of_logs")
    if (
        isinstance(reported_count, bool)
        or not isinstance(reported_count, int)
        or reported_count != row_count
    ):
        raise ValueError("Verification report log count does not match metadata.")
    if report.get("embedding_shape") != list(embeddings.shape):
        raise ValueError("Verification report embedding_shape does not match tensor.")
    if report.get("embedding_finite") is not True:
        raise ValueError("Verification report must confirm finite embeddings.")
    if report.get("official_test_evaluation_performed") is not False:
        raise ValueError(
            "Verification report must confirm no official test evaluation."
        )


def _validate_report_configuration(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    for field in ("model_name", "max_length", "pooling"):
        if field not in report:
            raise ValueError(f"Verification report is missing {field}.")
        try:
            report_value = (
                _positive_integer(report[field], field)
                if field == "max_length"
                else report[field].strip()
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"Verification report has invalid {field}.") from exc
        if report_value != config[field]:
            raise ValueError(
                f"Verification report {field} does not match embedding metadata."
            )
    for field in (
        "tokenizer_name", "tokenizer_revision", "model_revision",
        "embedding_model", "embedding_revision",
    ):
        if field in report:
            value = report[field]
            value = value.strip() if isinstance(value, str) and value.strip() else None
            if field not in config or value != config[field]:
                raise ValueError(
                    f"Verification report {field} does not match embedding metadata."
                )


def _validate_processed_alignment(
    metadata_rows: Sequence[Mapping[str, Any]],
    processed_logs: Sequence[Mapping[str, Any]],
) -> None:
    for index, (metadata, processed) in enumerate(zip(metadata_rows, processed_logs)):
        for field in ("unit_id", "cycle", "source_split"):
            metadata_value = str(metadata.get(field, "") or "")
            processed_value = str(processed.get(field, "") or "")
            if metadata_value != processed_value:
                raise ValueError(
                    f"NLP metadata/processed-log mismatch at row {index}: {field}."
                )
