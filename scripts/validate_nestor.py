"""Validate the existing NLP pipeline on the NIST Nestor excavator dataset.

This is an independent NLP validation run, not FD001 sensor-text fusion.
The source dataset and its records are preserved.
"""

import csv
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.preprocessing import preprocess_logs
from nlp.ner import extract_entities
from nlp.severity import score_severity
from nlp.embeddings import TextEmbedder, MODEL_NAME, MAX_LENGTH

INPUT = ROOT / "data" / "external" / "nestor" / "excavators.csv"
DATA_DIR = ROOT / "artifacts" / "data" / "nestor_nlp"
EMBED_DIR = ROOT / "artifacts" / "embeddings" / "nestor_nlp"


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(batch_size=8):
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "BscStartDate",
            "Asset",
            "OriginalShorttext",
            "PMType",
            "Cost",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing source columns: {sorted(missing)}")
        source_rows = list(reader)

    if not source_rows:
        raise ValueError("Dataset has no records.")

    # Deterministic row IDs; source order is preserved.
    logs = []
    for index, row in enumerate(source_rows):
        logs.append({
            "log_id": f"nestor-row-{index + 1:06d}",
            "text": row["OriginalShorttext"],
            "source_dataset": "nist_nestor_excavators",
            "source_row_number": index + 1,
            "asset": row["Asset"],
            "maintenance_date": row["BscStartDate"],
            "pm_type": row["PMType"],
            "cost": row["Cost"],
        })

    processed = preprocess_logs(logs)
    entity_rows = []
    severity_rows = []

    for row in processed:
        original = row["original_text"]
        original = original if isinstance(original, str) else ""

        entities = extract_entities(original, log_id=row["log_id"])
        entity_rows.extend(entities)

        severity = score_severity(row["normalized_text"])
        row["severity_score"] = severity["severity_score"]
        row["severity_label"] = severity["severity_label"]
        row["severity_matched_terms"] = json.dumps(
            severity["matched_terms"], ensure_ascii=False
        )
        row["severity_explanation"] = severity["explanation"]

        severity_rows.append({
            "log_id": row["log_id"],
            "asset": row["asset"],
            "maintenance_date": row["maintenance_date"],
            "pm_type": row["pm_type"],
            "severity_score": severity["severity_score"],
            "severity_label": severity["severity_label"],
            "matched_terms": json.dumps(
                severity["matched_terms"], ensure_ascii=False
            ),
            "explanation": severity["explanation"],
        })

    # Embed normalized text in batches; no model training.
    embedder = TextEmbedder()
    tensors = []
    texts = [row["normalized_text"] for row in processed]

    for start in range(0, len(texts), batch_size):
        tensors.append(embedder.encode(texts[start:start + batch_size]))

    embeddings = torch.cat(tensors, dim=0)

    expected_shape = (len(source_rows), 768)
    if tuple(embeddings.shape) != expected_shape:
        raise RuntimeError(
            f"Expected embedding shape {expected_shape}; "
            f"got {tuple(embeddings.shape)}"
        )
    if not torch.isfinite(embeddings).all().item():
        raise RuntimeError("Embeddings contain NaN or infinite values.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    processed_fields = list(dict.fromkeys(
        key for row in processed for key in row
    ))
    write_csv(DATA_DIR / "processed_logs.csv", processed, processed_fields)

    with (DATA_DIR / "entities.jsonl").open("w", encoding="utf-8") as f:
        for entity in entity_rows:
            f.write(json.dumps(entity, ensure_ascii=False) + "\n")

    write_csv(
        DATA_DIR / "severity.csv",
        severity_rows,
        [
            "log_id",
            "asset",
            "maintenance_date",
            "pm_type",
            "severity_score",
            "severity_label",
            "matched_terms",
            "explanation",
        ],
    )

    torch.save(embeddings, EMBED_DIR / "embeddings.pt")

    metadata = []
    for index, row in enumerate(processed):
        metadata.append({
            "row_index": index,
            "log_id": row["log_id"],
            "asset": row["asset"],
            "maintenance_date": row["maintenance_date"],
            "source_row_number": row["source_row_number"],
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "pooling": "attention-mask mean pooling",
            "embedding_dim": 768,
        })

    write_csv(
        EMBED_DIR / "embedding_metadata.csv",
        metadata,
        [
            "row_index",
            "log_id",
            "asset",
            "maintenance_date",
            "source_row_number",
            "model_name",
            "max_length",
            "pooling",
            "embedding_dim",
        ],
    )

    severity_counts = Counter(
        str(row["severity_score"]) for row in severity_rows
    )
    entity_counts = Counter(entity["label"] for entity in entity_rows)

    report = {
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "dataset": "NIST Nestor excavator maintenance-work-order toy dataset",
        "dataset_type": "external_maintenance_text_nlp_validation",
        "input_file": str(INPUT.relative_to(ROOT)),
        "number_of_records": len(source_rows),
        "number_of_processed_records": len(processed),
        "invalid_or_empty_text_records": sum(
            row["text_status"] != "ok" for row in processed
        ),
        "number_of_entities": len(entity_rows),
        "entity_counts_by_label": dict(entity_counts),
        "severity_score_distribution": dict(severity_counts),
        "severity_is_heuristic_not_ground_truth": True,
        "pm_type_is_severity_label": False,
        "embedding_shape": list(embeddings.shape),
        "embedding_finite": bool(torch.isfinite(embeddings).all().item()),
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "pooling": "attention-mask mean pooling",
        "training_performed": False,
        "sensor_text_fusion_performed": False,
        "official_fd001_test_evaluated": False,
        "limitations": [
            "Nestor is documented as a toy maintenance-work-order dataset.",
            "Rule-based entity extraction has no labelled NER evaluation here.",
            "Severity scores are heuristic and are not ground-truth labels.",
            "PMType represents maintenance type, not severity.",
            "No matching sensor telemetry or FD001 cycle is provided.",
            "Source duplicates are preserved; records are not silently removed.",
        ],
    }

    (DATA_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Nestor NLP validation completed.")
    print(f"Records: {len(source_rows)}")
    print(f"Entities: {len(entity_rows)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"All embeddings finite: {report['embedding_finite']}")
    print(f"Python version: {report['environment']['python_version']}")
    print(f"PyTorch version: {report['environment']['torch_version']}")
    print(
        "Transformers version: "
        f"{report['environment']['transformers_version']}"
    )
    print(f"Data outputs: {DATA_DIR}")
    print(f"Embedding outputs: {EMBED_DIR}")
    print(f"Report: {DATA_DIR / 'verification_report.json'}")


if __name__ == "__main__":
    main()