
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


class DatasetValidationError(RuntimeError):
    """Raised when repository dataset artifacts fail validation."""


def _require_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()

    if not path.is_relative_to(root):
        raise DatasetValidationError(f"Path escapes repository: {relative_path}")

    if not path.is_file():
        raise DatasetValidationError(f"Required file is missing: {relative_path}")

    return path


def _read_json(root: Path, relative_path: str) -> dict[str, Any]:
    path = _require_file(root, relative_path)

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"Cannot read JSON report {relative_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise DatasetValidationError(f"Expected JSON object: {relative_path}")

    return data


def _check_array(
    root: Path,
    relative_path: str,
    expected_shape: tuple[int, ...],
) -> dict[str, Any]:
    path = _require_file(root, relative_path)

    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(
            f"Cannot load array {relative_path}: {exc}"
        ) from exc

    if array.shape != expected_shape:
        raise DatasetValidationError(
            f"{relative_path}: expected shape {expected_shape}, got {array.shape}"
        )

    if not np.issubdtype(array.dtype, np.number):
        raise DatasetValidationError(f"{relative_path}: expected numeric data")

    if not np.isfinite(array).all():
        raise DatasetValidationError(f"{relative_path}: contains non-finite values")

    return {
        "path": relative_path,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_values": True,
    }


def _read_csv(root: Path, relative_path: str) -> list[dict[str, str]]:
    path = _require_file(root, relative_path)

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise DatasetValidationError(
            f"Cannot read CSV {relative_path}: {exc}"
        ) from exc


def _validate_fd001(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    expected_shapes = {
        "X_train": (14207, 30, 17),
        "X_val": (3524, 30, 17),
        "X_test": (100, 30, 17),
        "y_train": (14207,),
        "y_val": (3524,),
        "y_test": (100,),
    }

    arrays = {}
    for name, shape in expected_shapes.items():
        arrays[name] = _check_array(
            root,
            f"artifacts/data/fd001_lstm_baseline/{name}.npy",
            shape,
        )

    for name, shape in expected_shapes.items():
        report_key = f"{name}_shape"
        if report.get(report_key) != list(shape):
            raise DatasetValidationError(
                f"FD001 report has incorrect {report_key}: {report.get(report_key)}"
            )

    if report.get("finite_values") is not True:
        raise DatasetValidationError("FD001 report does not confirm finite values")

    scaler = report.get("scaler_report")
    if not isinstance(scaler, dict):
        raise DatasetValidationError("FD001 scaler_report is missing or invalid")

    if scaler.get("scaler") != "StandardScaler":
        raise DatasetValidationError("Unexpected FD001 scaler")

    if scaler.get("fit_scope") != "training_engine_rows_only":
        raise DatasetValidationError("FD001 scaler fit scope is not training-only")

    if scaler.get("fit_rows") != 16527 or scaler.get("fit_feature_count") != 17:
        raise DatasetValidationError("FD001 scaler fit metadata is inconsistent")

    raw_files = [
        "Data/train_FD001.txt",
        "Data/test_FD001.txt",
        "Data/RUL_FD001.txt",
    ]
    for relative_path in raw_files:
        _require_file(root, relative_path)

    embedding_path = "artifacts/embeddings/fd001_lstm_baseline/sensor_embeddings.npy"
    metadata_path = (
        "artifacts/embeddings/fd001_lstm_baseline/sensor_embedding_metadata.csv"
    )

    metadata = _read_csv(root, metadata_path)
    if len(metadata) != 17731:
        raise DatasetValidationError(
            f"Sensor embedding metadata expected 17731 rows, got {len(metadata)}"
        )

    embeddings = _check_array(root, embedding_path, (17731, 64))

    required_columns = {
        "unit_id", "window_start_cycle", "window_end_cycle",
        "label_cycle", "label_rul", "source_split",
        "embedding_index", "embedding_source",
    }
    if not required_columns.issubset(metadata[0].keys()):
        raise DatasetValidationError("Sensor embedding metadata columns are incomplete")

    return {
        "arrays": arrays,
        "sensor_embeddings": embeddings,
        "sensor_embedding_metadata_rows": len(metadata),
        "official_test_evaluation": "not_run",
    }


def _validate_nlp(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    if report.get("dataset_type") != "synthetic_fixture":
        raise DatasetValidationError("NLP fixture is not identified as synthetic")

    input_file = report.get("input_file")
    if not isinstance(input_file, str) or not input_file:
        raise DatasetValidationError("NLP fixture input_file is missing")

    input_path = _require_file(root, input_file)
    processed_path = _require_file(
        root, "artifacts/data/fd001_nlp_fixture/processed_logs.csv"
    )
    entities_path = _require_file(
        root, "artifacts/data/fd001_nlp_fixture/entities.jsonl"
    )
    severity_path = _require_file(
        root, "artifacts/data/fd001_nlp_fixture/severity.csv"
    )

    embedding_path = _require_file(
        root, "artifacts/embeddings/fd001_nlp_fixture/embeddings.pt"
    )
    metadata_path = "artifacts/embeddings/fd001_nlp_fixture/embedding_metadata.csv"

    try:
        embeddings = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise DatasetValidationError(
            f"Cannot safely load NLP embeddings: {exc}"
        ) from exc

    if not isinstance(embeddings, torch.Tensor):
        raise DatasetValidationError("NLP embeddings are not a tensor")

    if tuple(embeddings.shape) != (5, 768):
        raise DatasetValidationError(
            f"Expected NLP embedding shape (5, 768), got {tuple(embeddings.shape)}"
        )

    if not torch.isfinite(embeddings).all().item():
        raise DatasetValidationError("NLP embeddings contain non-finite values")

    metadata = _read_csv(root, metadata_path)
    if len(metadata) != 5:
        raise DatasetValidationError(
            f"Expected 5 NLP embedding metadata rows, got {len(metadata)}"
        )

    required_columns = {
        "row_index", "log_id", "unit_id", "cycle",
        "source_split", "model_name", "max_length",
        "pooling", "embedding_dim",
    }
    if not required_columns.issubset(metadata[0].keys()):
        raise DatasetValidationError("NLP embedding metadata columns are incomplete")

    processed = _read_csv(root, str(processed_path.relative_to(root)))
    severity = _read_csv(root, str(severity_path.relative_to(root)))

    if len(processed) != 5 or len(severity) != 5:
        raise DatasetValidationError("NLP processed/severity row count is not 5")

    entity_count = 0
    try:
        with entities_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
                    entity_count += 1
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Invalid NLP entities JSONL: {exc}") from exc

    if entity_count != 16:
        raise DatasetValidationError(
            f"Expected 16 NLP entities, got {entity_count}"
        )

    if report.get("number_of_logs") != 5:
        raise DatasetValidationError("NLP report log count is inconsistent")

    if report.get("number_of_entities") != 16:
        raise DatasetValidationError("NLP report entity count is inconsistent")

    if report.get("embedding_shape") != [5, 768]:
        raise DatasetValidationError("NLP report embedding shape is inconsistent")

    if report.get("embedding_finite") is not True:
        raise DatasetValidationError("NLP report does not confirm finite embeddings")

    if report.get("official_test_evaluation_performed") is not False:
        raise DatasetValidationError("Unexpected NLP official-test evaluation claim")

    return {
        "input_file": input_file,
        "embedding_path": str(embedding_path.relative_to(root)),
        "embedding_shape": [5, 768],
        "embedding_finite": True,
        "metadata_rows": len(metadata),
        "processed_logs": len(processed),
        "severity_rows": len(severity),
        "entity_count": entity_count,
        "classification": "synthetic_fixture",
    }


def build_inventory(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()

    benchmark_report_path = "artifacts/data/fd001_lstm_baseline/data_report.json"
    fixture_report_path = "artifacts/data/fd001_nlp_fixture/verification_report.json"

    benchmark_report = _read_json(root, benchmark_report_path)
    fixture_report = _read_json(root, fixture_report_path)

    benchmark_evidence = _validate_fd001(root, benchmark_report)
    fixture_evidence = _validate_nlp(root, fixture_report)

    return {
        "inventory_version": 2,
        "validation_status": "PASS",
        "datasets": [
            {
                "descriptor": {
                    "namespace": "fd001_benchmark",
                    "name": "NASA C-MAPSS FD001",
                    "classification": "benchmark",
                    "source": "NASA C-MAPSS FD001 files in Data/",
                },
                "suitability": {
                    "sensor_time_series": True,
                    "nlp": False,
                    "genuine_multimodal": False,
                },
                "evidence": benchmark_evidence,
            },
            {
                "descriptor": {
                    "namespace": "fd001_nlp_fixture",
                    "name": "FD001 synthetic technician-log fixture",
                    "classification": "synthetic_fixture",
                    "source": "Synthetic technician-log fixture; not genuine paired FD001 text",
                },
                "suitability": {
                    "sensor_time_series": False,
                    "nlp": True,
                    "genuine_multimodal": False,
                },
                "evidence": fixture_evidence,
            },
        ],
        "policy": {
            "cross_dataset_pairing_inferred": False,
            "synthetic_fixture_is_genuine_technician_data": False,
            "benchmark_and_fixture_kept_separate": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate repository datasets and build a local inventory."
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for writing the JSON inventory report.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    try:
        report = build_inventory(root)
        rendered = json.dumps(report, indent=2, ensure_ascii=False)

        if args.report:
            output_path = args.report
            if not output_path.is_absolute():
                output_path = root / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
            print(f"Dataset inventory written to: {output_path}")
        else:
            print(rendered)

        return 0

    except DatasetValidationError as exc:
        print(f"DATASET VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
