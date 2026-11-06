"""Synthetic NLP artifact manifest generation and intake tests."""

import csv
import hashlib
import json
import inspect

import pytest
import torch

from scripts import build_nlp_artifacts as builder
from tests.test_nlp_fusion_pipeline import _write_package
from training.nlp_artifact_validation import validate_nlp_artifact_package
from scripts import verify_nlp


PAYLOADS = (
    "embeddings/embeddings.pt",
    "embeddings/embedding_metadata.csv",
    "data/processed_logs.csv",
    "data/entities.jsonl",
    "data/severity.csv",
    "data/verification_report.json",
)


def _stub_nlp(monkeypatch):
    monkeypatch.setattr(
        builder,
        "preprocess_logs",
        lambda rows: [
            {**row, "original_text": row["text"], "normalized_text": row["text"].lower()}
            for row in rows
        ],
    )
    monkeypatch.setattr(builder, "extract_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        builder,
        "score_severity",
        lambda _text: {
            "severity_score": 0,
            "severity_label": "unknown",
            "matched_terms": [],
            "explanation": "fixture test",
        },
    )

    class Embedder:
        def encode(self, texts):
            return torch.zeros((len(texts), 768), dtype=torch.float32)

    monkeypatch.setattr(builder, "TextEmbedder", Embedder)


def _build_fixture(tmp_path, monkeypatch):
    _stub_nlp(monkeypatch)
    input_path = tmp_path / "fixture.csv"
    with input_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["log_id", "unit_id", "cycle", "text", "source_split"],
        )
        writer.writeheader()
        writer.writerow({
            "log_id": "fixture-log-1",
            "unit_id": "1",
            "cycle": "1",
            "text": "Synthetic bearing note",
            "source_split": "fixture",
        })
    package_dir = tmp_path / "package"
    builder.build(
        input_path,
        package_dir / "data",
        package_dir / "embeddings",
        dataset_type="synthetic_fixture",
    )
    return package_dir


def _create_payload_files(package_dir):
    payload_paths = [package_dir / relative for relative in PAYLOADS]
    for path in payload_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
    return payload_paths


def _write_input_csv(path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["log_id", "unit_id", "cycle", "text", "source_split"],
        )
        writer.writeheader()
        writer.writerow({
            "log_id": "fixture-log-1",
            "unit_id": "1",
            "cycle": "1",
            "text": "Synthetic bearing note",
            "source_split": "fixture",
        })


def test_default_paths_are_isolated_under_fixture_package_root():
    assert builder.DEFAULT_PACKAGE_DIR == builder.ROOT / "artifacts" / "fd001_nlp_fixture"
    assert builder.DEFAULT_DATA_DIR == builder.DEFAULT_PACKAGE_DIR / "data"
    assert builder.DEFAULT_EMBEDDING_DIR == builder.DEFAULT_PACKAGE_DIR / "embeddings"
    assert inspect.signature(builder.build).parameters["dataset_type"].default is inspect.Parameter.empty


def test_build_requires_explicit_synthetic_opt_in_and_rejects_other_types(tmp_path):
    input_path = tmp_path / "fixture.csv"
    _write_input_csv(input_path)
    package_dir = tmp_path / "package"

    with pytest.raises(TypeError, match="dataset_type"):
        builder.build(input_path, package_dir / "data", package_dir / "embeddings")
    with pytest.raises(ValueError, match="only for synthetic_fixture"):
        builder.build(
            input_path,
            package_dir / "data",
            package_dir / "embeddings",
            dataset_type="approved_real",
        )


def test_manifest_rejects_missing_duplicate_unexpected_and_mislocated_payloads(tmp_path):
    package_dir = tmp_path / "package"
    complete_paths = _create_payload_files(package_dir)

    with pytest.raises(ValueError, match="Missing manifest payload"):
        builder.write_artifact_manifest(package_dir, complete_paths[:-1])
    complete_paths[-1].unlink()
    with pytest.raises(ValueError, match="regular file"):
        builder.write_artifact_manifest(package_dir, complete_paths)
    complete_paths[-1].write_bytes(b"payload")
    with pytest.raises(ValueError, match="Duplicate manifest payload"):
        builder.write_artifact_manifest(package_dir, [*complete_paths, complete_paths[0]])
    with pytest.raises(ValueError, match="Unexpected or incorrectly located"):
        builder.write_artifact_manifest(
            package_dir,
            [*complete_paths[:-1], package_dir / "data" / "unexpected.csv"],
        )
    with pytest.raises(ValueError, match="Unexpected or incorrectly located"):
        builder.write_artifact_manifest(
            package_dir,
            [*complete_paths[:-1], package_dir / "archive" / "verification_report.json"],
        )
    assert not (package_dir / "artifact_manifest.json").exists()


def test_manifest_writer_preserves_existing_manifest(tmp_path):
    package_dir = tmp_path / "package"
    payload_paths = _create_payload_files(package_dir)
    manifest_path = package_dir / "artifact_manifest.json"
    original_contents = '{"preserve": "existing manifest"}\n'
    manifest_path.write_text(original_contents, encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing artifact manifest"):
        builder.write_artifact_manifest(package_dir, payload_paths)

    assert manifest_path.read_text(encoding="utf-8") == original_contents


def test_existing_output_collision_is_rejected_before_writing(tmp_path, monkeypatch):
    _stub_nlp(monkeypatch)
    input_path = tmp_path / "fixture.csv"
    _write_input_csv(input_path)
    package_dir = tmp_path / "package"
    data_dir = package_dir / "data"
    embedding_dir = package_dir / "embeddings"
    data_dir.mkdir(parents=True)
    collision = data_dir / "processed_logs.csv"
    collision.write_text("unrelated existing content", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.build(
            input_path,
            data_dir,
            embedding_dir,
            dataset_type="synthetic_fixture",
        )

    assert collision.read_text(encoding="utf-8") == "unrelated existing content"
    assert not embedding_dir.exists()


def test_manifest_is_written_after_all_payloads_and_hashes_actual_bytes(
    tmp_path, monkeypatch
):
    _stub_nlp(monkeypatch)
    input_path = tmp_path / "fixture.csv"
    with input_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["log_id", "unit_id", "cycle", "text", "source_split"],
        )
        writer.writeheader()
        writer.writerow({
            "log_id": "fixture-log-1",
            "unit_id": "1",
            "cycle": "1",
            "text": "Synthetic bearing note",
            "source_split": "fixture",
        })

    package_dir = tmp_path / "package"
    original_writer = builder.write_artifact_manifest
    observed = {}

    def checking_writer(root, payload_paths):
        payload_paths = list(payload_paths)
        observed["all_exist_before_manifest"] = all(path.is_file() for path in payload_paths)
        original_writer(root, payload_paths)

    monkeypatch.setattr(builder, "write_artifact_manifest", checking_writer)
    builder.build(
        input_path,
        package_dir / "data",
        package_dir / "embeddings",
        dataset_type="synthetic_fixture",
    )

    assert observed["all_exist_before_manifest"] is True
    manifest_path = package_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_type"] == "synthetic_fixture"
    assert set(manifest["files"]) == set(PAYLOADS)
    for relative_path in PAYLOADS:
        entry = manifest["files"][relative_path]
        expected = hashlib.sha256((package_dir / relative_path).read_bytes()).hexdigest()
        assert entry == {"sha256": expected}


def test_generated_synthetic_package_passes_without_genuine_claim(
    tmp_path, monkeypatch
):
    package_dir = _build_fixture(tmp_path, monkeypatch)

    result = validate_nlp_artifact_package(package_dir)
    assert verify_nlp.main(package_dir) == 0

    assert result["status"] == "PASS"
    assert result["genuine_data_claim"] is False
    checks = {check["name"]: check["status"] for check in result["checks"]}
    assert checks["manifest_hashes"] == "PASS"
    assert checks["dataset_classification"] == "PASS"
    assert checks["authenticity"] == "BLOCKED"
    assert checks["source_provenance"] == "BLOCKED"
    assert checks["model_tokenizer_revisions"] == "BLOCKED"
    assert checks["timestamps"] == "BLOCKED"


def test_payload_tampering_fails_manifest_validation(tmp_path, monkeypatch):
    package_dir = _build_fixture(tmp_path, monkeypatch)
    with (package_dir / "data" / "severity.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    result = validate_nlp_artifact_package(package_dir)

    assert result["status"] == "FAIL"
    assert result["genuine_data_claim"] is False
    check = next(item for item in result["checks"] if item["name"] == "manifest_hashes")
    assert check["status"] == "FAIL"
    assert "severity.csv" in check["detail"]


def test_missing_and_malformed_manifests_are_rejected(tmp_path):
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    _write_package(missing_dir)
    (missing_dir / "artifact_manifest.json").unlink()

    missing = validate_nlp_artifact_package(missing_dir)
    assert missing["status"] == "BLOCKED"
    assert missing["genuine_data_claim"] is False

    malformed_dir = tmp_path / "malformed"
    malformed_dir.mkdir()
    _write_package(malformed_dir)
    (malformed_dir / "artifact_manifest.json").write_text("{bad json", encoding="utf-8")

    malformed = validate_nlp_artifact_package(malformed_dir)
    assert malformed["status"] == "FAIL"
    assert malformed["genuine_data_claim"] is False
