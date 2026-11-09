"""Synthetic-only tests for the reusable NLP package intake validator."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from training.nlp_artifact_validation import (
    ArtifactValidationLimits,
    _aware_timestamp,
    validate_nlp_artifact_package,
)


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "verify_nlp.py"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025-01-01T10:00:00Z", True),
        ("2025-01-01T10:00:00+05:30", True),
        ("2025-01-01T10:00:00-07:00", True),
        ("2025-01-01T10:00:00", False),
        ("not-a-timestamp", False),
        ("", False),
        (None, False),
    ],
)
def test_aware_timestamp_validation(value, expected):
    assert _aware_timestamp(value) is expected


def _package(tmp_path, *, include_hashes=True):
    rows = [{
        "row_index": "0", "log_id": "fixture-log-0", "unit_id": "2", "cycle": "5",
        "source_split": "fixture", "model_name": "fixture-model", "tokenizer_name": "fixture-tokenizer",
        "model_revision": "fixture-revision", "tokenizer_revision": "fixture-tokenizer-revision",
        "embedding_dim": "768", "source_system": "synthetic-test", "source_record_id": "fixture-source-0",
        "source_dataset": "synthetic-fixture", "source_dataset_version": "fixture-v1",
        "event_timestamp": "2025-01-01T10:00:00Z", "created_at": "2025-01-01T10:00:00Z",
        "ingested_at": "2025-01-01T10:00:00Z", "available_at": "2025-01-01T10:00:00Z",
    }]
    torch.save(torch.zeros((1, 768)), tmp_path / "embeddings.pt")
    for name in ("embedding_metadata.csv", "processed_logs.csv", "severity.csv"):
        fields = list(rows[0]) if name == "embedding_metadata.csv" else (
            ["log_id", "unit_id", "cycle", "source_split"] if name == "processed_logs.csv"
            else ["log_id", "severity_score"]
        )
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow({k: rows[0].get(k, "0.1") for k in fields})
    (tmp_path / "entities.jsonl").write_text(json.dumps({"log_id": "fixture-log-0", "entities": []}) + "\n", encoding="utf-8")
    (tmp_path / "verification_report.json").write_text(json.dumps({"embedding_shape": [1, 768], "number_of_logs": 1}), encoding="utf-8")
    files = ["embeddings.pt", "embedding_metadata.csv", "processed_logs.csv", "entities.jsonl", "severity.csv", "verification_report.json"]
    manifest = {"dataset_type": "synthetic_fixture"}
    if include_hashes:
        manifest["files"] = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in files}
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rows


def _status(report, name):
    return next(check["status"] for check in report["checks"] if check["name"] == name)


def test_valid_synthetic_package_checks_shape_alignment_fields_and_hashes(tmp_path):
    _package(tmp_path)
    report = validate_nlp_artifact_package(tmp_path)
    assert report["genuine_data_claim"] is False
    assert _status(report, "embedding_shape") == "PASS"
    assert _status(report, "row_index_coverage") == "PASS"
    assert _status(report, "processed_log_alignment") == "PASS"
    assert _status(report, "manifest_hashes") == "PASS"
    assert _status(report, "authenticity") == "BLOCKED"


def test_missing_package_files_are_reported_blocked(tmp_path):
    report = validate_nlp_artifact_package(tmp_path)
    assert report["status"] == "BLOCKED"
    assert _status(report, "required_files") == "BLOCKED"


def test_invalid_row_index_and_naive_timestamps_fail_without_inference(tmp_path):
    _package(tmp_path, include_hashes=False)
    path = tmp_path / "embedding_metadata.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0])
    rows[0]["row_index"] = "4"
    rows[0]["unit_id"] = ""
    rows[0]["available_at"] = "2025-01-01T10:00:00"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = validate_nlp_artifact_package(tmp_path)
    assert _status(report, "row_index_coverage") == "FAIL"
    assert _status(report, "unit_cycle") == "FAIL"
    assert _status(report, "timestamps") == "BLOCKED"
    assert _status(report, "manifest_hashes") == "BLOCKED"


def test_nan_embedding_and_hash_mismatch_fail(tmp_path):
    _package(tmp_path)
    torch.save(torch.full((1, 768), float("nan")), tmp_path / "embeddings.pt")
    report = validate_nlp_artifact_package(tmp_path)
    assert _status(report, "embedding_finite") == "FAIL"
    assert _status(report, "manifest_hashes") == "FAIL"


def test_inconsistent_processed_id_and_untrusted_classification_are_blocked(tmp_path):
    _package(tmp_path)
    path = tmp_path / "processed_logs.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0])
    rows[0]["log_id"] = "different-log-id"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report_path = tmp_path / "verification_report.json"
    report_path.write_text(json.dumps({"embedding_shape": [1, 768], "number_of_logs": 1, "dataset_type": "approved_real"}), encoding="utf-8")
    report = validate_nlp_artifact_package(tmp_path)
    assert _status(report, "processed_log_alignment") == "FAIL"
    assert _status(report, "dataset_classification") == "BLOCKED"
    assert report["genuine_data_claim"] is False


def test_synthetic_classification_requires_fixture_source_split(tmp_path):
    _package(tmp_path, include_hashes=False)
    (tmp_path / "verification_report.json").write_text(
        json.dumps({"dataset_type": "synthetic_fixture"}), encoding="utf-8"
    )
    for filename in ("embedding_metadata.csv", "processed_logs.csv"):
        path = tmp_path / filename
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
            fields = list(rows[0])
        for row in rows:
            row["source_split"] = "train"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    report = validate_nlp_artifact_package(tmp_path)
    assert _status(report, "synthetic_contract") == "FAIL"
    assert report["status"] == "FAIL"


def test_oversized_file_is_reported_as_fail(tmp_path):
    _package(tmp_path)
    report = validate_nlp_artifact_package(
        tmp_path, limits=ArtifactValidationLimits(max_file_bytes=1)
    )
    assert report["status"] == "FAIL"
    assert _status(report, "resource_limits") == "FAIL"


def test_excessive_csv_rows_are_reported_as_fail(tmp_path):
    _package(tmp_path)
    path = tmp_path / "processed_logs.csv"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write("fixture-log-1,2,6,fixture\n")
    report = validate_nlp_artifact_package(
        tmp_path, limits=ArtifactValidationLimits(max_csv_rows=1)
    )
    assert report["status"] == "FAIL"
    assert _status(report, "resource_limits") == "FAIL"
    assert "CSV row count" in report["blockers"][0]


def test_oversized_jsonl_record_is_reported_as_fail(tmp_path):
    _package(tmp_path)
    (tmp_path / "entities.jsonl").write_text(
        json.dumps({"log_id": "fixture-log-0", "entity_text": "too large"}) + "\n",
        encoding="utf-8",
    )
    report = validate_nlp_artifact_package(
        tmp_path, limits=ArtifactValidationLimits(max_jsonl_line_bytes=10)
    )
    assert report["status"] == "FAIL"
    assert _status(report, "resource_limits") == "FAIL"
    assert "JSONL line size" in report["blockers"][0]


def test_duplicate_json_object_keys_are_rejected_in_jsonl(tmp_path):
    _package(tmp_path, include_hashes=False)
    (tmp_path / "entities.jsonl").write_text(
        '{"log_id":"fixture-log-0","log_id":"shadowed","entities":[]}\n',
        encoding="utf-8",
    )
    report = validate_nlp_artifact_package(tmp_path)
    assert report["status"] == "FAIL"
    assert _status(report, "package_parse") == "FAIL"
    assert any("duplicate JSON object key" in blocker for blocker in report["blockers"])


def test_duplicate_json_object_keys_are_rejected_in_manifest(tmp_path):
    _package(tmp_path, include_hashes=False)
    (tmp_path / "artifact_manifest.json").write_text(
        '{"dataset_type":"synthetic_fixture","dataset_type":"approved_real"}',
        encoding="utf-8",
    )
    report = validate_nlp_artifact_package(tmp_path)
    assert report["status"] == "FAIL"
    assert _status(report, "package_parse") == "FAIL"
    assert any("duplicate JSON object key" in blocker for blocker in report["blockers"])


def test_oversized_tensor_is_reported_as_fail(tmp_path):
    _package(tmp_path)
    torch.save(torch.zeros((1, 768)), tmp_path / "embeddings.pt")
    report = validate_nlp_artifact_package(
        tmp_path, limits=ArtifactValidationLimits(max_tensor_elements=767)
    )
    assert report["status"] == "FAIL"
    assert _status(report, "resource_limits") == "FAIL"
    assert "tensor element count" in report["blockers"][0]


def test_tensor_limit_is_checked_before_torch_load(tmp_path, monkeypatch):
    _package(tmp_path)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("torch.load must not run for oversized tensor metadata")

    monkeypatch.setattr(torch, "load", unexpected_load)
    report = validate_nlp_artifact_package(
        tmp_path, limits=ArtifactValidationLimits(max_tensor_elements=767)
    )
    assert report["status"] == "FAIL"
    assert _status(report, "resource_limits") == "FAIL"


def test_symlinked_artifact_is_rejected(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _package(package)
    external = tmp_path / "external.pt"
    external.write_bytes((package / "embeddings.pt").read_bytes())
    (package / "embeddings.pt").unlink()
    try:
        os.symlink(external, package / "embeddings.pt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    report = validate_nlp_artifact_package(package)
    assert report["status"] == "FAIL"
    assert _status(report, "artifact_paths") == "FAIL"


@pytest.mark.parametrize("option", [
    "--max-file-bytes", "--max-csv-rows", "--max-jsonl-line-bytes",
    "--max-jsonl-records", "--max-tensor-elements",
])
def test_cli_rejects_non_positive_limits(option):
    result = subprocess.run(
        [sys.executable, str(_script_path()), option, "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert "Traceback" not in result.stderr


def test_report_cannot_overwrite_input_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    _package(package)
    from scripts.verify_nlp import main

    with pytest.raises(ValueError, match="outside the input package"):
        main(package, package / "verification_report.json")


def test_malformed_json_is_reported_as_fail(tmp_path):
    _package(tmp_path)
    (tmp_path / "verification_report.json").write_text("{not-json", encoding="utf-8")
    report = validate_nlp_artifact_package(tmp_path)
    assert report["status"] == "FAIL"
    assert _status(report, "package_parse") == "FAIL"
