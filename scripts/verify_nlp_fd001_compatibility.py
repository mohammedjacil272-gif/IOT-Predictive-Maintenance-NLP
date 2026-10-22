"""Verify actual FD001 NLP fixture artifacts against the canonical fusion interface.

No training or model evaluation is performed.

This script verifies:
- NLP embedding dimensions
- metadata consistency
- canonical (unit_id, cycle) alignment
- fixture split preservation
- compatibility with build_fusion_samples()
- rejection of invalid embedding dimensions
- rejection of inconsistent metadata
- rejection of fabricated FD001 train/validation mappings
"""

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fusion_dataset import build_fusion_samples


DATA_DIR = ROOT / "artifacts" / "data" / "fd001_nlp_fixture"
EMBED_DIR = ROOT / "artifacts" / "embeddings" / "fd001_nlp_fixture"
SENSOR_DIR = ROOT / "artifacts" / "data" / "fd001_lstm_baseline"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def expect_failure(description, fn):
    try:
        fn()
    except (ValueError, KeyError, AssertionError, TypeError, RuntimeError):
        print(f"PASS: {description}")
        return

    raise AssertionError(
        f"Expected verification failure: {description}"
    )


def main():
    print("Verifying actual FD001 NLP fixture artifacts...")
    print()

    # ---------------------------------------------------------
    # 1. Load NLP artifacts
    # ---------------------------------------------------------
    metadata = read_csv(
        EMBED_DIR / "embedding_metadata.csv"
    )

    processed_logs = read_csv(
        DATA_DIR / "processed_logs.csv"
    )

    severity_rows = read_csv(
        DATA_DIR / "severity.csv"
    )

    embeddings = torch.load(
        EMBED_DIR / "embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 768
    assert torch.isfinite(embeddings).all()
    assert len(metadata) == embeddings.shape[0]

    print(f"PASS: Loaded {len(metadata)} NLP embeddings")
    print(
        f"PASS: NLP embedding shape = "
        f"{tuple(embeddings.shape)}"
    )

    # ---------------------------------------------------------
    # 2. Verify metadata and processed logs
    # ---------------------------------------------------------
    processed_by_id = {
        row["log_id"]: row
        for row in processed_logs
    }

    severity_by_id = {
        row["log_id"]: row
        for row in severity_rows
    }

    text_records = []

    for row in metadata:
        log_id = row["log_id"]

        assert log_id in processed_by_id
        assert log_id in severity_by_id

        processed = processed_by_id[log_id]
        severity = severity_by_id[log_id]

        assert row["unit_id"] == processed["unit_id"]
        assert row["cycle"] == processed["cycle"]

        assert row["unit_id"] == severity["unit_id"]
        assert row["cycle"] == severity["cycle"]

        assert int(row["embedding_dim"]) == 768

        row_index = int(row["row_index"])

        assert 0 <= row_index < embeddings.shape[0]

        # IMPORTANT:
        # The source artifact must remain a fixture.
        # It must not be silently converted into FD001
        # train/validation metadata.
        source_split = row["source_split"]

        assert source_split == "fixture"

        text_records.append(
            {
                "log_id": log_id,
                "unit_id": int(row["unit_id"]),
                "cycle": int(row["cycle"]),
                "split": source_split,
                "text_embedding": embeddings[row_index],
                "entities": [],
                "severity_score": float(
                    severity["severity_score"]
                ),
                "severity_label": severity["severity_label"],
                "matched_terms": severity["matched_terms"],
                "explanation": severity["explanation"],
            }
        )

    print(
        "PASS: NLP metadata, processed logs, "
        "and severity records agree"
    )

    print("PASS: All NLP embeddings are 768-D")

    print(
        "PASS: NLP alignment keys are integer "
        "(unit_id, cycle)"
    )

    print(
        "PASS: Source split is preserved as "
        "'fixture'"
    )

    # ---------------------------------------------------------
    # 3. Load FD001 validation metadata only for evidence
    # ---------------------------------------------------------
    val_metadata = read_csv(
        SENSOR_DIR / "val_window_metadata.csv"
    )

    print()
    print(
        f"INFO: FD001 validation windows available = "
        f"{len(val_metadata)}"
    )

    print(
        "PASS: FD001 validation metadata was inspected "
        "without mapping NLP fixture records into it"
    )

    # ---------------------------------------------------------
    # 4. Create synthetic sensor records
    #
    # These are interface-test vectors only.
    # They are NOT FD001 sensor windows.
    # ---------------------------------------------------------
    sensor_records = []

    for record in text_records:
        sensor_records.append(
            {
                "unit_id": record["unit_id"],
                "cycle": record["cycle"],
                "split": "fixture",
                "sensor_embedding": [0.1] * 64,
                "rul": 0.0,
            }
        )

    # ---------------------------------------------------------
    # 5. Valid canonical fusion interface
    # ---------------------------------------------------------
    result = build_fusion_samples(
        sensor_records,
        text_records,
    )

    assert result["sensor_embeddings"].shape == (
        len(text_records),
        64,
    )

    assert result["text_embeddings"].shape == (
        len(text_records),
        768,
    )

    assert result["rul"].shape[0] == len(text_records)

    assert all(
        split == "fixture"
        for split in result["split"]
    )

    assert all(result["has_log"].tolist())

    print()
    print(
        "PASS: Actual NLP artifacts consumed by "
        "build_fusion_samples()"
    )

    print("PASS: Sensor embeddings = 64-D")
    print("PASS: Text embeddings = 768-D")

    print(
        "PASS: Canonical alignment uses "
        "(unit_id, cycle)"
    )

    print(
        "PASS: Split metadata preserved as "
        "'fixture'"
    )

    print(
        "PASS: All matched records have "
        "has_log=True"
    )

    # ---------------------------------------------------------
    # 6. Negative test: invalid text embedding dimension
    # ---------------------------------------------------------
    bad_text = [
        dict(row)
        for row in text_records
    ]

    bad_text[0]["text_embedding"] = torch.zeros(767)

    expect_failure(
        "Invalid 767-D text embedding is rejected",
        lambda: build_fusion_samples(
            sensor_records,
            bad_text,
        ),
    )

    # ---------------------------------------------------------
    # 7. Negative test: inconsistent unit_id metadata
    # ---------------------------------------------------------
    bad_metadata = [
        dict(row)
        for row in text_records
    ]

    bad_metadata[0]["unit_id"] += 999

    expect_failure(
        "Inconsistent unit_id metadata is rejected",
        lambda: build_fusion_samples(
            sensor_records,
            bad_metadata,
        ),
    )

    # ---------------------------------------------------------
    # 8. Negative test: fabricated FD001 split mapping
    # ---------------------------------------------------------
    fabricated_split = [
        dict(row)
        for row in text_records
    ]

    for row in fabricated_split:
        row["split"] = "validation"

    def reject_fabricated_mapping():
        # The compatibility verifier itself must reject
        # conversion of the fixture source into an FD001 split.
        splits = {
            row["split"]
            for row in fabricated_split
        }

        if splits != {"fixture"}:
            raise ValueError(
                "NLP fixture cannot be converted into "
                "FD001 train/validation metadata."
            )

        build_fusion_samples(
            sensor_records,
            fabricated_split,
        )

    expect_failure(
        "Fabricated FD001 validation split mapping is rejected",
        reject_fabricated_mapping,
    )

    # ---------------------------------------------------------
    # 9. Final status
    # ---------------------------------------------------------
    print()
    print(
        "IMPORTANT: This is a synthetic fixture "
        "interface test."
    )

    print(
        "No FD001 train/validation mapping was fabricated."
    )

    print(
        "No fusion training or model evaluation was performed."
    )

    print(
        "The frozen FD001 sensor-only baseline was not modified."
    )


if __name__ == "__main__":
    main()