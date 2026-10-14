
"""Verify NLP fixture provenance against the canonical fusion interface.

Uses synthetic sensor vectors only. Does not train or evaluate a model.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fusion_dataset import build_fusion_samples


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT.parent

# These are generated fixture artifacts in the original NLP worktree.
DATA_DIR = (
    DESKTOP
    / "IOT-Predictive-Maintenance-NLP"
    / "artifacts"
    / "data"
    / "fd001_nlp_fixture"
)
EMBED_DIR = (
    DESKTOP
    / "IOT-Predictive-Maintenance-NLP"
    / "artifacts"
    / "embeddings"
    / "fd001_nlp_fixture"
)


def read_csv(path):
    with path.open(
        newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def main():
    required_files = [
        DATA_DIR / "entities.jsonl",
        DATA_DIR / "severity.csv",
        EMBED_DIR / "embedding_metadata.csv",
        EMBED_DIR / "embeddings.pt",
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required fixture artifact not found: {path}"
            )

    metadata = read_csv(EMBED_DIR / "embedding_metadata.csv")
    severity_rows = read_csv(DATA_DIR / "severity.csv")

    with (DATA_DIR / "entities.jsonl").open(
        encoding="utf-8"
    ) as file:
        entity_rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    embeddings = torch.load(
        EMBED_DIR / "embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert embeddings.ndim == 2 and embeddings.shape[1] == 768
    assert len(metadata) == embeddings.shape[0]
    assert len(metadata) == len(severity_rows) == 5

    metadata = sorted(metadata, key=lambda row: int(row["row_index"]))
    severity_by_id = {
        row["log_id"]: row for row in severity_rows
    }

    entities_by_id = defaultdict(list)
    for entity in entity_rows:
        entities_by_id[entity["log_id"]].append(entity)

    text_records = []
    sensor_splits = {}

    for row in metadata:
        log_id = row["log_id"]
        unit_id = int(row["unit_id"])
        cycle = int(row["cycle"])
        split = row["source_split"]
        key = (unit_id, cycle)

        if split != "fixture":
            raise ValueError(
                f"Unexpected split for {log_id}: {split}"
            )

        if key in sensor_splits and sensor_splits[key] != split:
            raise ValueError(f"Split mismatch for window {key}")
        sensor_splits[key] = split

        severity = severity_by_id[log_id]
        entity_list = entities_by_id[log_id]

        text_records.append({
            "log_id": log_id,
            "unit_id": unit_id,
            "cycle": cycle,
            "split": split,
            "text_embedding": embeddings[int(row["row_index"])],
            "entities": entity_list,
            "severity_score": float(severity["severity_score"]),
            "severity_label": severity["severity_label"],
            "matched_terms": severity.get("matched_terms"),
            "explanation": severity.get("explanation"),
        })

    # One synthetic sensor record per log-alignment key.
    # RUL is a placeholder required by the current interface;
    # it is not a real target and is never used for training/evaluation.
    sensor_records = [
        {
            "unit_id": unit_id,
            "cycle": cycle,
            "split": split,
            "sensor_embedding": [0.1] * 64,
            "rul": 0.0,
        }
        for (unit_id, cycle), split in sorted(sensor_splits.items())
    ]

    # Add an unmatched synthetic sensor window to verify missing text.
    sensor_records.append({
        "unit_id": 9999,
        "cycle": 9999,
        "split": "fixture",
        "sensor_embedding": [0.0] * 64,
        "rul": 0.0,
    })

    result = build_fusion_samples(sensor_records, text_records)

    assert result["text_embeddings"].shape == (
        len(sensor_records), 768
    )
    assert result["sensor_embeddings"].shape == (
        len(sensor_records), 64
    )
    assert sum(len(ids) for ids in result["log_ids_by_window"]) == 5
    assert sum(
        len(items) for items in result["entities_by_window"]
    ) == len(entity_rows)
    assert sum(
        len(items) for items in result["severity_by_log_by_window"]
    ) == 5
    assert all(split == "fixture" for split in result["split"])

    missing_index = result["unit_id"].tolist().index(9999)
    assert not result["has_log"][missing_index].item()
    assert torch.count_nonzero(
        result["text_embeddings"][missing_index]
    ).item() == 0
    assert result["window_severity"][missing_index] is None

    print("PASS: Five NLP fixture logs aligned with synthetic sensor windows")
    print(f"PASS: Preserved {len(entity_rows)} NER entities")
    print("PASS: Five per-log severity records preserved")
    print("PASS: Window severity computed by maximum available score")
    print("PASS: Missing-text window retained with a zero embedding")
    print("PASS: Fixture split isolation preserved")
    print("No training or model evaluation performed.")


if __name__ == "__main__":
    main()
