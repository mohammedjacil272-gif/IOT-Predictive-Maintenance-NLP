
"""Build NLP artifacts from a technician-log CSV."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# Make project-root imports work when this script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.preprocessing import preprocess_logs
from nlp.ner import extract_entities
from nlp.severity import score_severity
from nlp.embeddings import TextEmbedder, MODEL_NAME, MAX_LENGTH


def read_logs(input_path):
    """Read CSV logs and validate required columns."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"log_id", "text"}
        columns = set(reader.fieldnames or [])
        missing = required - columns

        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        logs = list(reader)

    if not logs:
        raise ValueError("Input CSV contains no data rows.")

    ids = [row["log_id"] for row in logs]
    if any(not value or not value.strip() for value in ids):
        raise ValueError("Every log must have a non-empty log_id.")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate log_id values found in input CSV.")

    return logs


def write_csv(path, rows, fieldnames):
    """Write dictionaries to a CSV file."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build(
    input_path,
    data_dir,
    embedding_dir,
    batch_size=8,
    dataset_type="synthetic_fixture",
):
    """Build row-aligned NLP artifacts and preserve per-log provenance."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    logs = read_logs(input_path)
    print(f"Loaded {len(logs)} logs from {input_path}")

    # 1. Normalize text while preserving original input fields.
    processed = preprocess_logs(logs)

    # 2. Extract entities from ORIGINAL text to preserve character offsets.
    entity_rows = []

    for row in processed:
        original = row.get("original_text")
        original = original if isinstance(original, str) else ""

        extracted_entities = extract_entities(
            original,
            log_id=row["log_id"],
        )

        # Preserve the source log and its alignment metadata for each entity.
        for entity in extracted_entities:
            entity.update({
                "unit_id": row.get("unit_id", ""),
                "cycle": row.get("cycle", ""),
                "source_split": row.get("source_split", ""),
            })
            entity_rows.append(entity)

    # 3. Score severity using normalized text.
    severity_rows = []

    for row in processed:
        result = score_severity(row["normalized_text"])

        row["severity_score"] = result["severity_score"]
        row["severity_label"] = result["severity_label"]
        row["severity_matched_terms"] = json.dumps(
            result["matched_terms"], ensure_ascii=False
        )
        row["severity_explanation"] = result["explanation"]

        severity_rows.append({
            "log_id": row["log_id"],
            "unit_id": row.get("unit_id", ""),
            "cycle": row.get("cycle", ""),
            "source_split": row.get("source_split", ""),
            "severity_score": result["severity_score"],
            "severity_label": result["severity_label"],
            "matched_terms": json.dumps(
                result["matched_terms"], ensure_ascii=False
            ),
            "explanation": result["explanation"],
        })

    # 4. Generate frozen DistilBERT embeddings in batches.
    texts = [row["normalized_text"] for row in processed]
    embedder = TextEmbedder()
    batches = []

    for start in range(0, len(texts), batch_size):
        batches.append(embedder.encode(texts[start:start + batch_size]))

    embeddings = torch.cat(batches, dim=0)

    if embeddings.shape != (len(logs), 768):
        raise RuntimeError(
            f"Unexpected embedding shape: {tuple(embeddings.shape)}"
        )

    if not torch.isfinite(embeddings).all().item():
        raise RuntimeError("Embeddings contain NaN or infinite values.")

    # 5. Create output directories only after validation and embedding.
    data_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save processed logs, including severity fields.
    processed_fields = list(dict.fromkeys(
        key for row in processed for key in row
    ))
    write_csv(
        data_dir / "processed_logs.csv",
        processed,
        processed_fields,
    )

    # 7. Save entities as JSON Lines, one entity per line.
    with (data_dir / "entities.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for entity in entity_rows:
            file.write(json.dumps(entity, ensure_ascii=False) + "\n")

    # 8. Save per-log severity, preserving identifiers and evidence.
    severity_fields = [
        "log_id",
        "unit_id",
        "cycle",
        "source_split",
        "severity_score",
        "severity_label",
        "matched_terms",
        "explanation",
    ]
    write_csv(
        data_dir / "severity.csv",
        severity_rows,
        severity_fields,
    )

    # 9. Save the text embedding tensor and row-aligned metadata.
    torch.save(embeddings, embedding_dir / "embeddings.pt")

    metadata_rows = []
    for index, row in enumerate(processed):
        metadata_rows.append({
            "row_index": index,
            "log_id": row["log_id"],
            "unit_id": row.get("unit_id", ""),
            "cycle": row.get("cycle", ""),
            "source_split": row.get("source_split", ""),
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "pooling": "attention-mask mean pooling",
            "embedding_dim": 768,
        })

    metadata_fields = [
        "row_index",
        "log_id",
        "unit_id",
        "cycle",
        "source_split",
        "model_name",
        "max_length",
        "pooling",
        "embedding_dim",
    ]
    write_csv(
        embedding_dir / "embedding_metadata.csv",
        metadata_rows,
        metadata_fields,
    )

    # 10. Save a verification report with explicit dataset classification.
    report = {
        "input_file": str(input_path),
        "dataset_type": dataset_type,
        "number_of_logs": len(logs),
        "number_of_entities": len(entity_rows),
        "embedding_shape": list(embeddings.shape),
        "embedding_finite": bool(
            torch.isfinite(embeddings).all().item()
        ),
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "pooling": "attention-mask mean pooling",
        "severity_distribution": {
            str(score): sum(
                row["severity_score"] == score
                for row in severity_rows
            )
            for score in range(5)
        },
        "official_test_evaluation_performed": False,
        "limitations": [
            "Input data is a synthetic fixture."
            if dataset_type == "synthetic_fixture"
            else "Dataset classification was supplied by the caller.",
            "Severity values are heuristic, not ground truth.",
            "Rule-based NER has not been evaluated on labeled real logs.",
        ],
    }

    with (data_dir / "verification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nPipeline completed successfully.")
    print(f"Dataset type: {dataset_type}")
    print(f"Logs: {len(logs)}")
    print(f"Entities: {len(entity_rows)}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"All embeddings finite: {report['embedding_finite']}")
    print(f"Data outputs: {data_dir}")
    print(f"Embedding outputs: {embedding_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="tests/fixtures/technician_logs.csv",
        help="Path to technician-log CSV",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--data-dir",
        default="artifacts/data/fd001_nlp_fixture",
        help="Directory for processed logs, entities, severity, and reports",
    )
    parser.add_argument(
        "--embedding-dir",
        default="artifacts/embeddings/fd001_nlp_fixture",
        help="Directory for embeddings and embedding metadata",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic_fixture", "approved_real"),
        default="synthetic_fixture",
        help="Dataset classification recorded in the report",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    data_dir = Path(args.data_dir)
    embedding_dir = Path(args.embedding_dir)

    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    if not embedding_dir.is_absolute():
        embedding_dir = ROOT / embedding_dir

    build(
        input_path,
        data_dir,
        embedding_dir,
        args.batch_size,
        args.dataset_type,
    )
