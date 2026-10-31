import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from scripts import build_nlp_artifacts, verify_fusion_provenance, verify_nlp
from training.fusion_dataset import build_fusion_samples


class DeterministicFixtureEmbedder:
    """Replace only remote model inference with deterministic fixture vectors."""

    def encode(self, texts):
        values = [sum(map(ord, text)) % 1000 / 1000 for text in texts]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(1).expand(-1, 768).clone()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_synthetic_nlp_artifacts_validate_and_flow_through_fusion_adapter(
    tmp_path, monkeypatch, request
):
    original_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(original_torch_threads))

    fixture_path = Path(__file__).parent / "fixtures" / "technician_logs.csv"
    data_dir = tmp_path / "data"
    embedding_dir = tmp_path / "embeddings"
    monkeypatch.setattr(build_nlp_artifacts, "TextEmbedder", DeterministicFixtureEmbedder)

    build_nlp_artifacts.build(
        fixture_path,
        data_dir,
        embedding_dir,
        batch_size=2,
        dataset_type="synthetic_fixture",
    )

    monkeypatch.setattr(verify_nlp, "DATA_DIR", data_dir)
    monkeypatch.setattr(verify_nlp, "EMBED_DIR", embedding_dir)
    verify_nlp.main()

    monkeypatch.setattr(verify_fusion_provenance, "DATA_DIR", data_dir)
    monkeypatch.setattr(verify_fusion_provenance, "EMBED_DIR", embedding_dir)
    verify_fusion_provenance.main()

    metadata = read_csv(embedding_dir / "embedding_metadata.csv")
    processed = read_csv(data_dir / "processed_logs.csv")
    severity_rows = read_csv(data_dir / "severity.csv")
    entities_by_log = defaultdict(list)
    with (data_dir / "entities.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entity = json.loads(line)
                entities_by_log[entity["log_id"]].append(entity)
    embeddings = torch.load(
        embedding_dir / "embeddings.pt", map_location="cpu", weights_only=True
    )

    processed_by_id = {row["log_id"]: row for row in processed}
    severity_by_id = {row["log_id"]: row for row in severity_rows}
    metadata_by_id = {row["log_id"]: row for row in metadata}
    for log_id, entities in entities_by_log.items():
        embedding_metadata = metadata_by_id[log_id]
        expected = {
            "unit_id": int(embedding_metadata["unit_id"]),
            "cycle": int(embedding_metadata["cycle"]),
            "source_split": embedding_metadata["source_split"],
        }
        for entity in entities:
            verify_fusion_provenance.validate_optional_provenance(
                entity, expected, f"integration entity for {log_id}"
            )

    text_records = []
    sensor_keys = set()
    for row in metadata:
        log_id = row["log_id"]
        unit_id, cycle = int(row["unit_id"]), int(row["cycle"])
        sensor_keys.add((unit_id, cycle))
        assert processed_by_id[log_id]["original_text"]
        assert processed_by_id[log_id]["normalized_text"]
        assert all(entity["log_id"] == log_id for entity in entities_by_log[log_id])
        text_records.append({
            "log_id": log_id,
            "unit_id": unit_id,
            "cycle": cycle,
            "split": row["source_split"],
            "namespace": "synthetic_fixture",
            "text_embedding": embeddings[int(row["row_index"])],
            "entities": entities_by_log[log_id],
            "severity_score": float(severity_by_id[log_id]["severity_score"]),
            "severity_label": severity_by_id[log_id]["severity_label"],
        })

    sensor_records = [
        {
            "unit_id": unit_id,
            "cycle": cycle,
            "split": "fixture",
            "namespace": "synthetic_fixture",
            "sensor_embedding": [0.25] * 64,
            "rul": 1.0,
        }
        for unit_id, cycle in sorted(sensor_keys)
    ]
    sensor_records.append({
        "unit_id": 9999,
        "cycle": 1,
        "split": "fixture",
        "namespace": "synthetic_fixture",
        "sensor_embedding": [0.0] * 64,
        "rul": 1.0,
    })

    samples = build_fusion_samples(
        sensor_records,
        text_records,
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
    )

    assert embeddings.shape == (5, 768)
    assert torch.isfinite(embeddings).all()
    assert samples["sensor_embeddings"].shape == (6, 64)
    assert samples["text_embeddings"].shape == (6, 768)
    assert torch.isfinite(samples["text_embeddings"]).all()
    assert samples["split"] == ("fixture",) * 6
    assert samples["has_log"].tolist() == [True] * 5 + [False]
    assert torch.count_nonzero(samples["text_embeddings"][-1]).item() == 0
    assert samples["log_ids_by_window"][-1] == []
    assert samples["entities_by_window"][-1] == []
    assert samples["severity_by_log_by_window"][-1] == []

    for record in text_records:
        key = (record["unit_id"], record["cycle"])
        index = list(zip(samples["unit_id"].tolist(), samples["cycle"].tolist())).index(key)
        assert record["log_id"] in samples["log_ids_by_window"][index]
        assert any(
            entity["log_id"] == record["log_id"]
            and entity["unit_id"] == record["unit_id"]
            and entity["cycle"] == record["cycle"]
            and entity["split"] == record["split"]
            for entity in samples["entities_by_window"][index]
        )
        assert any(
            severity["log_id"] == record["log_id"]
            and severity["severity_label"] == record["severity_label"]
            for severity in samples["severity_by_log_by_window"][index]
        )

    malformed = dict(text_records[0])
    malformed["entities"] = [dict(text_records[0]["entities"][0], log_id="unknown-log")]
    with pytest.raises(ValueError, match="Entity provenance mismatch.*log_id"):
        build_fusion_samples(
            sensor_records,
            [malformed, *text_records[1:]],
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
        )
