import csv
import json
import os
from pathlib import Path

import pytest
import torch

from training.fusion_dataset import build_fusion_samples


def test_fixture_artifacts_compatible_with_fusion_interface():
    configured_root = os.environ.get("IOT_NLP_ARTIFACTS")

    if configured_root:
        root = Path(configured_root)
    else:
        root = (
            Path.home()
            / "Desktop"
            / "IOT-Predictive-Maintenance-NLP"
            / "artifacts"
        )

    data_dir = root / "data" / "fd001_nlp_fixture"
    embedding_dir = root / "embeddings" / "fd001_nlp_fixture"

    required_files = [
        data_dir / "severity.csv",
        data_dir / "entities.jsonl",
        embedding_dir / "embedding_metadata.csv",
        embedding_dir / "embeddings.pt",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        pytest.skip(
            "Fixture artifacts unavailable; set IOT_NLP_ARTIFACTS. "
            f"Missing: {missing}"
        )

    tensor = torch.load(
        embedding_dir / "embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )

    with (embedding_dir / "embedding_metadata.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        metadata = list(csv.DictReader(f))

    with (data_dir / "severity.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        severity_rows = list(csv.DictReader(f))

    with (data_dir / "entities.jsonl").open(encoding="utf-8") as f:
        entity_rows = [json.loads(line) for line in f if line.strip()]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (5, 768)
    assert tensor.is_floating_point()
    assert torch.isfinite(tensor).all()
    assert len(metadata) == len(severity_rows) == 5
    assert {row["source_split"] for row in metadata} == {"fixture"}
    assert {row["source_split"] for row in severity_rows} == {"fixture"}

    metadata_by_id = {row["log_id"]: row for row in metadata}
    severity_by_id = {row["log_id"]: row for row in severity_rows}
    assert len(metadata_by_id) == 5
    assert metadata_by_id.keys() == severity_by_id.keys()

    entities_by_id = {}
    for entity in entity_rows:
        entities_by_id.setdefault(entity["log_id"], []).append(entity)
    assert set(entities_by_id) == set(metadata_by_id)

    text_records = []
    sensor_records = []

    for log_id in sorted(metadata_by_id):
        meta = metadata_by_id[log_id]
        severity = severity_by_id[log_id]

        assert meta["source_split"] == "fixture"
        assert severity["unit_id"] == meta["unit_id"]
        assert severity["cycle"] == meta["cycle"]
        assert severity["source_split"] == meta["source_split"]

        unit_id = int(meta["unit_id"])
        cycle = int(meta["cycle"])

        text_records.append({
            "log_id": log_id,
            "unit_id": unit_id,
            "cycle": cycle,
            "split": "fixture",
            "text_embedding": tensor[int(meta["row_index"])],
            "entities": entities_by_id[log_id],
            "severity_score": float(severity["severity_score"]),
            "severity_label": severity["severity_label"],
            "matched_terms": json.loads(severity["matched_terms"]),
            "explanation": severity["explanation"],
        })

        # Synthetic interface-only placeholder; never use for training.
        sensor_records.append({
            "unit_id": unit_id,
            "cycle": cycle,
            "split": "fixture",
            "sensor_embedding": [0.0] * 64,
            "rul": 0.0,
        })

    result = build_fusion_samples(sensor_records, text_records)

    assert result["sensor_embeddings"].shape == (5, 64)
    assert result["text_embeddings"].shape == (5, 768)
    assert result["has_log"].tolist() == [True] * 5
    assert set(result["split"]) == {"fixture"}
    assert result["rul"].tolist() == [0.0] * 5
    assert sorted(
        log_id
        for group in result["log_ids_by_window"]
        for log_id in group
    ) == sorted(metadata_by_id)
    assert len(result["entities_by_window"]) == 5
    assert len(result["severity_by_log_by_window"]) == 5