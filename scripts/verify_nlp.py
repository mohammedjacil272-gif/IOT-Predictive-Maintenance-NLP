
"""Verify synthetic NLP artifacts and embedding-to-log alignment."""

import csv
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data" / "fd001_nlp_fixture"
EMBED_DIR = ROOT / "artifacts" / "embeddings" / "fd001_nlp_fixture"

EXPECTED_DIM = 768
EXPECTED_MODEL = "distilbert-base-uncased"
EXPECTED_MAX_LENGTH = 128
EXPECTED_POOLING = "attention-mask mean pooling"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_csv(path):
    require(path.exists(), f"File does not exist: {path.name}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames is not None, f"{path.name} has no header")
        return list(reader)


def validate_unique_ids(rows, source_name):
    ids = [str(row.get("log_id") or "").strip() for row in rows]

    require(
        all(ids),
        f"{source_name}: log_id values must be non-empty",
    )
    require(
        len(ids) == len(set(ids)),
        f"{source_name}: duplicate log_id values found",
    )

    return ids


def validate_alignment(processed, metadata, tensor):
    """Validate embedding rows against their processed-log metadata."""

    require(bool(processed), "Processed logs are empty")
    require(bool(metadata), "Embedding metadata is empty")
    require(isinstance(tensor, torch.Tensor), "Embedding file is not a tensor")
    require(tensor.ndim == 2, "Embedding tensor must be two-dimensional")
    require(tensor.shape[1] == EXPECTED_DIM, "Embedding dimension must be 768")
    require(bool(torch.isfinite(tensor).all()), "Embeddings contain non-finite values")

    tensor_rows = tensor.shape[0]

    require(
        tensor_rows == len(metadata),
        "Tensor row count does not match metadata row count",
    )
    require(
        len(metadata) == len(processed),
        "Metadata row count does not match processed-log row count",
    )

    processed_ids = validate_unique_ids(processed, "Processed logs")
    metadata_ids = validate_unique_ids(metadata, "Embedding metadata")

    row_indices = []
    for row in metadata:
        try:
            row_indices.append(int(row["row_index"]))
        except (KeyError, TypeError, ValueError):
            raise AssertionError("Invalid or missing row_index") from None

    require(
        row_indices == list(range(tensor_rows)),
        "Metadata row indexes must be sequential from zero",
    )

    require(
        metadata_ids == processed_ids,
        "Embedding metadata log_id order does not match processed logs",
    )

    for index, (meta, log) in enumerate(zip(metadata, processed)):
        for field in ("unit_id", "cycle", "source_split"):
            require(
                meta.get(field, "") == log.get(field, ""),
                f"Row {index}: {field} does not match processed log",
            )

    for index, row in enumerate(metadata):
        try:
            max_length = int(row.get("max_length", ""))
            embedding_dim = int(row.get("embedding_dim", ""))
        except (TypeError, ValueError):
            raise AssertionError(
                f"Row {index}: invalid embedding configuration values"
            ) from None

        require(
            row.get("model_name") == EXPECTED_MODEL,
            f"Row {index}: unexpected embedding model",
        )
        require(
            max_length == EXPECTED_MAX_LENGTH,
            f"Row {index}: unexpected maximum token length",
        )
        require(
            row.get("pooling") == EXPECTED_POOLING,
            f"Row {index}: unexpected pooling configuration",
        )
        require(
            embedding_dim == EXPECTED_DIM,
            f"Row {index}: unexpected metadata embedding dimension",
        )

    require(
        all(row.get("source_split") == "fixture" for row in processed),
        "Processed records must all be labelled fixture",
    )


def main():
    print("Verifying synthetic NLP artifacts and alignment...\n")

    processed = read_csv(DATA_DIR / "processed_logs.csv")
    metadata = read_csv(EMBED_DIR / "embedding_metadata.csv")

    tensor_path = EMBED_DIR / "embeddings.pt"
    require(tensor_path.exists(), "Embedding tensor does not exist")

    tensor = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=True,
    )

    validate_alignment(processed, metadata, tensor)

    report_path = DATA_DIR / "verification_report.json"
    require(report_path.exists(), "Verification report does not exist")

    with report_path.open("r", encoding="utf-8-sig") as file:
        report = json.load(file)

    require(
        report.get("dataset_type") == "synthetic_fixture",
        "Report does not identify synthetic fixture data",
    )
    require(
        report.get("number_of_logs") == len(processed),
        "Report log count does not match",
    )
    require(
        report.get("embedding_shape") == list(tensor.shape),
        "Report embedding shape does not match",
    )
    require(
        report.get("official_test_evaluation_performed") is False,
        "Report must confirm that official test evaluation was not performed",
    )

    print("PASS: embedding-to-log alignment")
    print("PASS: metadata and embedding configuration")
    print("PASS: synthetic-only classification")
    print("PASS: verification report")
    print("\nAll synthetic NLP alignment checks passed.")
    print(f"Verified logs: {len(processed)}")
    print(f"Embedding shape: {tuple(tensor.shape)}")
    print("Dataset classification: SYNTHETIC-ONLY")


if __name__ == "__main__":
    main()
