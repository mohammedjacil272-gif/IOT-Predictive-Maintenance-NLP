
"""Verify NLP fixture provenance against the canonical fusion interface.

Uses synthetic sensor vectors only.
Does not train or evaluate a model.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.fusion_dataset import build_fusion_samples


def build_fixture_samples(sensor_records, text_records):
    """Call the interface with explicit synthetic provenance and policies."""
    return build_fusion_samples(
        [dict(row, namespace="synthetic_fixture") for row in sensor_records],
        [dict(row, namespace="synthetic_fixture") for row in text_records],
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
    )


# Resolve artifacts from the current repository worktree.
DATA_DIR = ROOT / "artifacts" / "data" / "fd001_nlp_fixture"
EMBED_DIR = ROOT / "artifacts" / "embeddings" / "fd001_nlp_fixture"


def read_csv(path):
    """Read a CSV file using UTF-8 encoding."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def normalize_optional_integer(record, field, context):
    """Normalize optional integer metadata and reject invalid values."""
    value = record.get(field)

    if value is None or str(value).strip() == "":
        # Remove blank metadata so the canonical interface can populate it
        # from the validated log metadata rather than reject an empty string.
        record.pop(field, None)
        return None

    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field} in {context}: {value!r}"
        ) from exc

    record[field] = normalized
    return normalized


def validate_optional_provenance(record, expected, context):
    """Validate provenance whenever the source supplies the field."""
    for field in ("unit_id", "cycle"):
        actual = normalize_optional_integer(record, field, context)

        if actual is not None and actual != int(expected[field]):
            raise ValueError(
                f"Provenance mismatch in {context}: {field}; "
                f"expected {expected[field]}, got {actual}"
            )

    split = record.get("source_split")

    if split is not None and str(split).strip():
        if split != expected["source_split"]:
            raise ValueError(
                f"Provenance mismatch in {context}: source_split; "
                f"expected {expected['source_split']!r}, got {split!r}"
            )


def index_by_log_id(rows, source_name):
    """Index records by log_id while rejecting missing or duplicate IDs."""
    indexed = {}

    for row in rows:
        log_id = row.get("log_id")

        if not log_id:
            raise ValueError(f"Empty log_id in {source_name}.")

        if log_id in indexed:
            raise ValueError(
                f"Duplicate log_id {log_id} in {source_name}."
            )

        indexed[log_id] = row

    return indexed


def main():
    required_files = [
        DATA_DIR / "entities.jsonl",
        DATA_DIR / "processed_logs.csv",
        DATA_DIR / "severity.csv",
        DATA_DIR / "verification_report.json",
        EMBED_DIR / "embedding_metadata.csv",
        EMBED_DIR / "embeddings.pt",
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required fixture artifact not found: {path}"
            )

    # Load the actual artifacts from this worktree.
    metadata = read_csv(EMBED_DIR / "embedding_metadata.csv")
    processed_rows = read_csv(DATA_DIR / "processed_logs.csv")
    severity_rows = read_csv(DATA_DIR / "severity.csv")

    with (DATA_DIR / "entities.jsonl").open(
        encoding="utf-8"
    ) as file:
        entity_rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    with (DATA_DIR / "verification_report.json").open(
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    embeddings = torch.load(
        EMBED_DIR / "embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Validate the embedding tensor.
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("Embeddings must be a PyTorch tensor.")

    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected embedding shape (n, 768), "
            f"got {tuple(embeddings.shape)}"
        )

    if not embeddings.is_floating_point():
        raise ValueError(
            "Embeddings must use a floating-point dtype."
        )

    if not torch.isfinite(embeddings).all().item():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            "Metadata row count does not match embedding count."
        )

    if len(metadata) != 5:
        raise ValueError(
            f"Expected five fixture metadata rows; got {len(metadata)}."
        )

    if len(processed_rows) != 5 or len(severity_rows) != 5:
        raise ValueError(
            "Expected five processed logs and five severity records."
        )

    # Validate the artifact report.
    if report.get("dataset_type") != "synthetic_fixture":
        raise ValueError(
            "Report must classify these artifacts as synthetic_fixture."
        )

    if report.get("official_test_evaluation_performed") is not False:
        raise ValueError(
            "Report must state that official test evaluation was not performed."
        )

    if report.get("embedding_shape") != list(embeddings.shape):
        raise ValueError(
            "Report embedding shape does not match the actual tensor."
        )

    if report.get("embedding_finite") is not True:
        raise ValueError(
            "Report does not confirm finite embeddings."
        )

    if report.get("number_of_logs") != 5:
        raise ValueError(
            "Report must record exactly five logs."
        )

    if report.get("number_of_entities") != len(entity_rows):
        raise ValueError(
            "Report entity count does not match entities.jsonl."
        )

    print("PASS: Loaded five synthetic fixture logs")
    print("PASS: Embeddings have shape (5, 768) and finite values")

    # Validate metadata row indices and identifiers.
    metadata.sort(key=lambda row: int(row["row_index"]))

    log_ids = [row["log_id"] for row in metadata]

    if any(not log_id for log_id in log_ids):
        raise ValueError(
            "Metadata contains an empty log_id."
        )

    if len(log_ids) != len(set(log_ids)):
        raise ValueError(
            "Duplicate log_id in embedding metadata."
        )

    row_indices = [int(row["row_index"]) for row in metadata]

    if sorted(row_indices) != list(range(len(metadata))):
        raise ValueError(
            "row_index values must uniquely cover every embedding row."
        )

    processed_by_id = index_by_log_id(
        processed_rows, "processed_logs.csv"
    )

    severity_by_id = index_by_log_id(
        severity_rows, "severity.csv"
    )

    expected_ids = set(log_ids)

    if set(processed_by_id) != expected_ids:
        raise ValueError(
            "Processed log IDs do not match embedding metadata IDs."
        )

    if set(severity_by_id) != expected_ids:
        raise ValueError(
            "Severity log IDs do not match embedding metadata IDs."
        )

    # Validate canonical alignment keys and fixture split isolation.
    metadata_by_id = {}
    sensor_splits = {}
    seen_keys = set()

    for row in metadata:
        log_id = row["log_id"]

        try:
            unit_id = int(row["unit_id"])
            cycle = int(row["cycle"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid unit_id or cycle for log {log_id}."
            ) from exc

        split = row.get("source_split")
        key = (unit_id, cycle)

        if unit_id <= 0 or cycle <= 0:
            raise ValueError(
                f"Invalid alignment key for {log_id}: {key}"
            )

        if split != "fixture":
            raise ValueError(
                f"Unexpected split for {log_id}: {split!r}"
            )

        if key in seen_keys:
            raise ValueError(
                f"Duplicate fixture alignment key: {key}"
            )

        seen_keys.add(key)

        row["unit_id"] = unit_id
        row["cycle"] = cycle
        metadata_by_id[log_id] = row
        sensor_splits[key] = split

        validate_optional_provenance(
            processed_by_id[log_id],
            row,
            f"processed log {log_id}",
        )

        validate_optional_provenance(
            severity_by_id[log_id],
            row,
            f"severity record {log_id}",
        )

    # Validate severity scores before building fusion records.
    for log_id, severity in severity_by_id.items():
        try:
            score = float(severity["severity_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid severity score for {log_id}."
            ) from exc

        if not math.isfinite(score) or not 0.0 <= score <= 4.0:
            raise ValueError(
                f"Severity score outside the expected 0–4 range "
                f"for {log_id}: {score}"
            )

    # Group entities by log ID and validate supplied provenance.
    entities_by_id = defaultdict(list)

    for entity in entity_rows:
        if not isinstance(entity, dict):
            raise ValueError(
                "Each entity record must be a JSON object."
            )

        log_id = entity.get("log_id")

        if log_id not in metadata_by_id:
            raise ValueError(
                f"Entity references unknown log_id: {log_id}"
            )

        expected = metadata_by_id[log_id]

        # CSV-derived identifiers may be strings. Normalize them to integers,
        # then reject actual mismatches rather than bypassing validation.
        validate_optional_provenance(
            entity,
            expected,
            f"entity for log {log_id}",
        )

        entities_by_id[log_id].append(entity)

    print(
        f"PASS: Provenance validated for {len(entity_rows)} entities"
    )

    # Build text records from the actual NLP artifacts.
    text_records = []

    for row in metadata:
        log_id = row["log_id"]
        severity = severity_by_id[log_id]

        text_records.append({
            "log_id": log_id,
            "unit_id": row["unit_id"],
            "cycle": row["cycle"],
            "split": row["source_split"],
            "text_embedding": embeddings[int(row["row_index"])],
            "entities": entities_by_id[log_id],
            "severity_score": float(severity["severity_score"]),
            "severity_label": severity["severity_label"],
            "matched_terms": severity.get("matched_terms"),
            "explanation": severity.get("explanation"),
        })

    # Synthetic sensor vectors test the interface only.
    # RUL is a placeholder, not a valid target.
    # No training or evaluation uses these records.
    sensor_records = [
        {
            "unit_id": unit_id,
            "cycle": cycle,
            "split": split,
            "sensor_embedding": [0.1] * 64,
            "rul": 0.0,
        }
        for (unit_id, cycle), split in sorted(sensor_splits.items())
    ]

    # Include one unmatched synthetic window to test missing-text handling.
    sensor_records.append({
        "unit_id": 9999,
        "cycle": 9999,
        "split": "fixture",
        "sensor_embedding": [0.0] * 64,
        "rul": 0.0,
    })

    # Verify compatibility through the canonical fusion interface.
    result = build_fixture_samples(sensor_records, text_records)

    expected_count = len(sensor_records)

    if result["text_embeddings"].shape != (expected_count, 768):
        raise AssertionError(
            "Unexpected text embedding output shape."
        )

    if result["sensor_embeddings"].shape != (expected_count, 64):
        raise AssertionError(
            "Unexpected sensor embedding output shape."
        )

    if sum(
        len(ids) for ids in result["log_ids_by_window"]
    ) != 5:
        raise AssertionError(
            "Expected exactly five aligned log records."
        )

    if sum(
        len(items) for items in result["entities_by_window"]
    ) != len(entity_rows):
        raise AssertionError(
            "Entity records were not preserved."
        )

    # Verify that each log's entities remain in its correct fusion window.
    expected_entities_by_log = {
        "L001": 5,
        "L002": 3,
        "L003": 3,
        "L004": 1,
        "L005": 4,
    }

    observed_entities_by_log = {}

    for window_log_ids, window_entities in zip(
        result["log_ids_by_window"],
        result["entities_by_window"],
    ):
        window_counts = {}

        for entity in window_entities:
            log_id = entity.get("log_id")

            if log_id is None:
                raise AssertionError(
                    "A preserved entity is missing its log_id."
                )

            window_counts[log_id] = window_counts.get(log_id, 0) + 1

        if set(window_counts) != set(window_log_ids):
            raise AssertionError(
                "Entity log IDs do not match their fusion window log IDs."
            )

        for log_id in window_log_ids:
            if log_id in observed_entities_by_log:
                raise AssertionError(
                    f"Log {log_id} appears in multiple fusion windows."
                )

            observed_entities_by_log[log_id] = window_counts.get(log_id, 0)

    if observed_entities_by_log != expected_entities_by_log:
        raise AssertionError(
            f"Per-log entity counts mismatch. "
            f"Expected: {expected_entities_by_log}; "
            f"observed: {observed_entities_by_log}"
        )

    print("PASS: Exact entity counts preserved in each fusion window")
    

    if sum(
        len(items) for items in result["severity_by_log_by_window"]
    ) != 5:
        raise AssertionError(
            "Expected five per-log severity records."
        )

    if not all(split == "fixture" for split in result["split"]):
        raise AssertionError(
            "Fixture split isolation was not preserved."
        )

    missing_index = result["unit_id"].tolist().index(9999)

    if result["has_log"][missing_index].item():
        raise AssertionError(
            "Unmatched window unexpectedly has a log."
        )

    if torch.count_nonzero(
        result["text_embeddings"][missing_index]
    ).item() != 0:
        raise AssertionError(
            "Missing-text window should have a zero text embedding."
        )

    if result["window_severity"][missing_index] is not None:
        raise AssertionError(
            "Missing-text window should not have window severity."
        )

    print("PASS: Canonical 64D sensor / 768D text interface verified")
    print("PASS: Five per-log severity records preserved")
    print("PASS: Missing-text window handled correctly")
    print("PASS: Fixture split isolation preserved")
    print(f"Artifacts validated from: {ROOT}")
    print("No training or model evaluation performed.")
    print("No official FD001 test evaluation performed.")


if __name__ == "__main__":
    main()
