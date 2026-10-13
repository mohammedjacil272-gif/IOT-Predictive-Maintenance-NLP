"""Metadata-based alignment for multimodal FD001 samples."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch


def build_fusion_samples(
    sensor_records: Sequence[Mapping],
    text_records: Sequence[Mapping],
) -> dict[str, torch.Tensor]:
    """Join sensor windows and text embeddings using unit_id and cycle.

    Sensor records require:
        unit_id, cycle, split, sensor_embedding, rul

    Text records require:
        log_id, unit_id, cycle, split, text_embedding

    Missing text is represented by a zero vector and has_log=False.
    Multiple logs for a window are averaged in deterministic log_id order.
    """

    sensor_by_key = {}

    for record in sensor_records:
        key = (record["unit_id"], record["cycle"])

        if key in sensor_by_key:
            raise ValueError(f"Duplicate sensor window: {key}")

        embedding = torch.as_tensor(
            record["sensor_embedding"], dtype=torch.float32
        )
        if embedding.shape != (64,):
            raise ValueError(f"Sensor embedding for {key} must have 64 values.")
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite sensor embedding for {key}.")

        rul = float(record["rul"])
        if not torch.isfinite(torch.tensor(rul)):
            raise ValueError(f"Non-finite RUL target for {key}.")

        sensor_by_key[key] = {
            "embedding": embedding,
            "rul": rul,
            "split": record["split"],
        }

    if not sensor_by_key:
        raise ValueError("No sensor records supplied.")

    text_by_key = defaultdict(list)
    seen_logs = set()
    seen_log_ids = set()

    for record in text_records:
        key = (record["unit_id"], record["cycle"])

        if key not in sensor_by_key:
            raise ValueError(f"Text record has no matching sensor window: {key}")

        if record["split"] != sensor_by_key[key]["split"]:
            raise ValueError(f"Split mismatch for {key}")

        log_id = str(record["log_id"]).strip()
        if not log_id:
            raise ValueError("Text record has a missing or empty log_id.")

        if log_id in seen_log_ids:
            raise ValueError(f"Duplicate log_id: {log_id}")

        seen_log_ids.add(log_id)
        log_key = (log_id, record["unit_id"], record["cycle"])

        embedding = torch.as_tensor(
            record["text_embedding"], dtype=torch.float32
        )
        if embedding.shape != (768,):
            raise ValueError(f"Text embedding for {key} must have 768 values.")
        if not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite text embedding for {key}.")

        text_by_key[key].append((str(record["log_id"]), embedding))

    keys = sorted(sensor_by_key)
    sensor_vectors = []
    text_vectors = []
    targets = []
    has_logs = []
    splits = []

    for key in keys:
        sensor = sensor_by_key[key]
        logs = sorted(text_by_key[key], key=lambda item: item[0])

        sensor_vectors.append(sensor["embedding"])
        targets.append(sensor["rul"])
        splits.append(sensor["split"])

        if logs:
            text_vectors.append(
                torch.stack([embedding for _, embedding in logs]).mean(dim=0)
            )
            has_logs.append(True)
        else:
            text_vectors.append(torch.zeros(768, dtype=torch.float32))
            has_logs.append(False)

    return {
        "unit_id": torch.tensor([key[0] for key in keys]),
        "cycle": torch.tensor([key[1] for key in keys]),
        "sensor_embeddings": torch.stack(sensor_vectors),
        "text_embeddings": torch.stack(text_vectors),
        "rul": torch.tensor(targets, dtype=torch.float32),
        "has_log": torch.tensor(has_logs, dtype=torch.bool),
        "split": tuple(splits),
    }
