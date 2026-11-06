
"""Build NLP artifacts from a technician-log CSV."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

# Make project-root imports work when this script is run directly.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = ROOT / "artifacts" / "fd001_nlp_fixture"
DEFAULT_DATA_DIR = DEFAULT_PACKAGE_DIR / "data"
DEFAULT_EMBEDDING_DIR = DEFAULT_PACKAGE_DIR / "embeddings"
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


def write_artifact_manifest(package_dir, payload_paths):
    """Write caller-asserted synthetic classification and payload byte hashes.

    The caller's synthetic classification is an assertion, not independent
    proof of source origin or genuine-data authenticity.
    """
    package_dir = Path(package_dir)
    expected_paths = {
        "embeddings/embeddings.pt",
        "embeddings/embedding_metadata.csv",
        "data/processed_logs.csv",
        "data/entities.jsonl",
        "data/severity.csv",
        "data/verification_report.json",
    }
    files = {}
    for path in payload_paths:
        path = Path(path)
        try:
            relative_path = path.relative_to(package_dir).as_posix()
        except ValueError as exc:
            raise ValueError("Manifest payload paths must be inside the package root.") from exc
        if relative_path in files:
            raise ValueError(f"Duplicate manifest payload path: {relative_path}")
        if relative_path not in expected_paths:
            raise ValueError(f"Unexpected or incorrectly located manifest payload: {relative_path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Manifest payload must be a regular file: {relative_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative_path] = {"sha256": digest}

    missing_paths = expected_paths - files.keys()
    if missing_paths:
        raise ValueError(f"Missing manifest payload paths: {sorted(missing_paths)}")

    manifest = {
        "dataset_type": "synthetic_fixture",
        "files": files,
    }
    manifest_path = package_dir / "artifact_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact manifest: {manifest_path}"
        )
    # Exclusive creation also protects against a manifest appearing after the
    # explicit preflight check above.
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def build(
    input_path,
    data_dir,
    embedding_dir,
    *,
    dataset_type,
    batch_size=8,
):
    """Build row-aligned NLP artifacts after caller asserts fixture origin.

    ``dataset_type='synthetic_fixture'`` is a caller assertion required to
    create a synthetic manifest; it does not independently prove data origin.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if dataset_type != "synthetic_fixture":
        raise ValueError(
            "Manifest generation is supported only for synthetic_fixture data."
        )

    data_dir = Path(data_dir)
    embedding_dir = Path(embedding_dir)
    if data_dir.name != "data" or embedding_dir.name != "embeddings":
        raise ValueError(
            "Artifact output directories must be package-root data/ and embeddings/."
        )
    if data_dir.parent.resolve() != embedding_dir.parent.resolve():
        raise ValueError(
            "data/ and embeddings/ must share one package root."
        )
    package_dir = data_dir.parent

    expected_outputs = [
        package_dir / "artifact_manifest.json",
        embedding_dir / "embeddings.pt",
        embedding_dir / "embedding_metadata.csv",
        data_dir / "processed_logs.csv",
        data_dir / "entities.jsonl",
        data_dir / "severity.csv",
        data_dir / "verification_report.json",
    ]
    collisions = [path for path in expected_outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing NLP artifact package files: "
            + ", ".join(str(path) for path in collisions)
        )

    logs = read_logs(input_path)
    if any(row.get("source_split") != "fixture" for row in logs):
        raise ValueError(
            "Synthetic artifact inputs must use source_split='fixture'."
        )
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
            "Synthetic-fixture classification is asserted by the caller and is not independent proof of data origin.",
            "No genuine-data authenticity or training claim is made.",
            "Severity values are heuristic, not ground truth.",
            "Rule-based NER has not been evaluated on labeled real logs.",
        ],
    }

    with (data_dir / "verification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # Hash the finalized payload bytes only after every package file has been
    # written successfully. The manifest itself is not self-hashed.
    write_artifact_manifest(package_dir, [
        embedding_dir / "embeddings.pt",
        embedding_dir / "embedding_metadata.csv",
        data_dir / "processed_logs.csv",
        data_dir / "entities.jsonl",
        data_dir / "severity.csv",
        data_dir / "verification_report.json",
    ])

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
        default=str(DEFAULT_DATA_DIR),
        help="Directory for processed logs, entities, severity, and reports",
    )
    parser.add_argument(
        "--embedding-dir",
        default=str(DEFAULT_EMBEDDING_DIR),
        help="Directory for embeddings and embedding metadata",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic_fixture",),
        required=True,
        help="Explicitly assert synthetic-fixture origin (not independently verified)",
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
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
    )
