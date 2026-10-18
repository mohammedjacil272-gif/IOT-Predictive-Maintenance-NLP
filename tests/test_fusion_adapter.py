import csv
import json

import pytest
import torch

from nlp.fusion_adapter import build_text_records, export_text_records


def create_artifacts(tmp_path, *, split="train", unit_id="1", cycle="10"):
    embedding_path = tmp_path / "embeddings.pt"
    metadata_path = tmp_path / "embedding_metadata.csv"
    severity_path = tmp_path / "severity.csv"
    entities_path = tmp_path / "entities.jsonl"

    torch.save(torch.ones((1, 768), dtype=torch.float32), embedding_path)

    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row_index", "log_id", "unit_id", "cycle", "source_split"
            ],
        )
        writer.writeheader()
        writer.writerow({
            "row_index": 0,
            "log_id": "L001",
            "unit_id": unit_id,
            "cycle": cycle,
            "source_split": split,
        })

    with severity_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "log_id", "unit_id", "cycle", "source_split",
                "severity_score", "severity_label",
                "matched_terms", "explanation",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "log_id": "L001",
            "unit_id": unit_id,
            "cycle": cycle,
            "source_split": split,
            "severity_score": 2,
            "severity_label": "medium",
            "matched_terms": '["moderate"]',
            "explanation": "Synthetic test only",
        })

    entities_path.write_text(
        json.dumps({
            "log_id": "L001",
            "unit_id": unit_id,
            "cycle": int(cycle),
            "source_split": split,
            "entity_text": "bearing",
            "label": "COMPONENT",
        }) + "\n",
        encoding="utf-8",
    )

    return embedding_path, metadata_path, severity_path, entities_path


def run_adapter(paths):
    return build_text_records(*paths)


def test_build_text_records_converts_artifacts(tmp_path):
    paths = create_artifacts(tmp_path)

    records = run_adapter(paths)

    assert len(records) == 1
    record = records[0]
    assert record["log_id"] == "L001"
    assert record["unit_id"] == 1
    assert record["cycle"] == 10
    assert record["split"] == "train"
    assert record["text_embedding"].shape == (768,)
    assert record["severity_score"] == 2.0
    assert record["matched_terms"] == ["moderate"]
    assert record["entities"][0]["label"] == "COMPONENT"


def test_rejects_fixture_split(tmp_path):
    paths = create_artifacts(tmp_path, split="fixture")

    with pytest.raises(ValueError, match="Unsupported split"):
        run_adapter(paths)


def test_rejects_missing_unit_id(tmp_path):
    paths = create_artifacts(tmp_path, unit_id="")

    with pytest.raises(ValueError, match="unit_id"):
        run_adapter(paths)


def test_rejects_severity_metadata_mismatch(tmp_path):
    paths = create_artifacts(tmp_path)

    severity_path = paths[2]
    rows = list(csv.DictReader(severity_path.open(encoding="utf-8")))
    rows[0]["cycle"] = "11"

    with severity_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="cycle mismatch"):
        run_adapter(paths)


def test_rejects_non_finite_embeddings(tmp_path):
    paths = create_artifacts(tmp_path)
    torch.save(torch.full((1, 768), float("nan")), paths[0])

    with pytest.raises(ValueError, match="NaN or Inf"):
        run_adapter(paths)

def test_export_text_records_round_trip(tmp_path):
    paths = create_artifacts(tmp_path)
    records = run_adapter(paths)
    output_path = tmp_path / "export" / "text_records.json"

    result_path = export_text_records(records, output_path)

    assert result_path == output_path
    assert output_path.is_file()

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(exported) == 1

    record = exported[0]
    assert record["log_id"] == "L001"
    assert record["unit_id"] == 1
    assert record["cycle"] == 10
    assert record["split"] == "train"
    assert len(record["text_embedding"]) == 768
    assert all(isinstance(value, (int, float)) for value in record["text_embedding"])
    assert all(torch.isfinite(torch.tensor(record["text_embedding"])))
    assert record["severity_score"] == 2.0
    assert record["matched_terms"] == ["moderate"]
    assert record["entities"][0]["label"] == "COMPONENT"


def test_export_rejects_invalid_embedding_shape(tmp_path):
    records = [{
        "log_id": "L001",
        "text_embedding": torch.ones(10),
    }]

    with pytest.raises(ValueError, match="shape"):
        export_text_records(records, tmp_path / "invalid.json")
def test_export_pipeline_rejects_synthetic_fixture_split(tmp_path):
    paths = create_artifacts(tmp_path, split="fixture")

    with pytest.raises(ValueError, match="Unsupported split"):
        records = run_adapter(paths)
        export_text_records(records, tmp_path / "text_records.json")
