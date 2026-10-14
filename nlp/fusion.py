
"""Prepare NLP embeddings for sensor-window fusion.

This module does not modify the sensor-only baseline.
It expects already-generated embeddings and explicit metadata.
"""

from collections import defaultdict
from typing import Any

import torch

EMBEDDING_DIM = 768
JOIN_KEYS = ("unit_id", "cycle")


def prepare_fusion(
    sensor_windows: list[dict[str, Any]],
    log_records: list[dict[str, Any]],
    embeddings: torch.Tensor,
) -> dict[str, Any]:
    """Align log embeddings with sensor windows using unit_id and cycle.

    Each log record must correspond to the embedding at the same input index.
    The caller must ensure that embeddings were generated in this exact order.
    """
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embeddings must be a torch.Tensor")

    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError("embeddings must have shape (n_logs, 768)")

    if embeddings.shape[0] != len(log_records):
        raise ValueError("Number of embeddings must match number of log records")

    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings contain non-finite values")

    # Build a sensor-window lookup and reject ambiguous keys.
    sensor_by_key = {}
    for window in sensor_windows:
        unit_id = window.get("unit_id")
        cycle = window.get("label_cycle")

        if unit_id is None or cycle is None:
            raise ValueError("Sensor windows require unit_id and label_cycle")

        key = (str(unit_id), str(cycle))
        if key in sensor_by_key:
            raise ValueError(f"Duplicate sensor window key: {key}")

        sensor_by_key[key] = window

    # Collect logs that have complete join metadata.
    grouped_logs = defaultdict(list)
    unmatched_logs = []
    unalignable_logs = []

    seen_log_ids = set()

    for index, log in enumerate(log_records):
        log_id = log.get("log_id")
        unit_id = log.get("unit_id")
        cycle = log.get("cycle")

        if log_id is None or str(log_id).strip() == "":
            raise ValueError(f"Log at index {index} has missing log_id")

        if str(log_id) in seen_log_ids:
            raise ValueError(f"Duplicate log_id: {log_id}")
        seen_log_ids.add(str(log_id))

        if unit_id is None or cycle is None:
            unalignable_logs.append(log_id)
            continue

        key = (str(unit_id), str(cycle))
        if key not in sensor_by_key:
            unmatched_logs.append(log_id)
            continue

        sensor_split = sensor_by_key[key].get("source_split")
        log_split = log.get("source_split")
        if sensor_split != log_split:
            raise ValueError(
                f"source_split mismatch for log_id={log_id}: "
                f"log={log_split!r}, sensor={sensor_split!r}"
            )

        grouped_logs[key].append((str(log_id), embeddings[index]))

    # Every sensor window is preserved, even if it has no log.
    output_rows = []
    output_embeddings = []

    for key, window in sensor_by_key.items():
        matching = sorted(grouped_logs.get(key, []), key=lambda item: item[0])

        if matching:
            stacked = torch.stack([embedding for _, embedding in matching])
            combined = stacked.mean(dim=0)
            has_log = True
            log_ids = [log_id for log_id, _ in matching]
        else:
            combined = torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
            has_log = False
            log_ids = []

        output_rows.append({
            "unit_id": window["unit_id"],
            "label_cycle": window["label_cycle"],
            "source_split": window.get("source_split"),
            "has_log": has_log,
            "log_ids": log_ids,
        })
        output_embeddings.append(combined)

    if output_embeddings:
        fused_text_embeddings = torch.stack(output_embeddings)
    else:
        fused_text_embeddings = torch.empty((0, EMBEDDING_DIM))

    return {
        "metadata": output_rows,
        "embeddings": fused_text_embeddings,
        "unmatched_log_ids": sorted(unmatched_logs),
        "unalignable_log_ids": sorted(unalignable_logs),
    }
