import pytest
import torch

from training.fusion_dataset import build_fusion_samples


def sensor(unit, cycle, split="train"):
    return {
        "unit_id": unit,
        "cycle": cycle,
        "split": split,
        "sensor_embedding": [float(unit)] * 64,
        "rul": float(100 - cycle),
    }


def log(log_id, unit, cycle, value, split="train"):
    return {
        "log_id": log_id,
        "unit_id": unit,
        "cycle": cycle,
        "split": split,
        "text_embedding": [float(value)] * 768,
    }


def test_alignment_and_missing_text():
    result = build_fusion_samples(
        [sensor(2, 1), sensor(1, 1)],
        [log("b", 1, 1, 4), log("a", 1, 1, 2)],
    )

    assert result["unit_id"].tolist() == [1, 2]
    assert result["has_log"].tolist() == [True, False]
    assert torch.all(result["text_embeddings"][0] == 3)
    assert torch.all(result["text_embeddings"][1] == 0)
    assert result["sensor_embeddings"].shape == (2, 64)
    assert result["text_embeddings"].shape == (2, 768)


def test_duplicate_sensor_window_rejected():
    with pytest.raises(ValueError, match="Duplicate sensor"):
        build_fusion_samples([sensor(1, 1), sensor(1, 1)], [])


def test_unmatched_text_rejected():
    with pytest.raises(ValueError, match="no matching sensor"):
        build_fusion_samples([sensor(1, 1)], [log("x", 9, 1, 2)])


def test_split_mismatch_rejected():
    with pytest.raises(ValueError, match="Split mismatch"):
        build_fusion_samples(
            [sensor(1, 1, "train")],
            [log("x", 1, 1, 2, "validation")],
        )

def test_duplicate_log_id_rejected_across_different_cycles():
    with pytest.raises(ValueError, match="Duplicate log_id"):
        build_fusion_samples(
            [sensor(1, 1), sensor(1, 2)],
            [
                log("same-id", 1, 1, 2),
                log("same-id", 1, 2, 3),
            ],
        )


def test_multiple_logs_are_aggregated_deterministically():
    result = build_fusion_samples(
        [sensor(1, 1)],
        [
            log("log-b", 1, 1, 6),
            log("log-a", 1, 1, 2),
        ],
    )

    assert result["has_log"].tolist() == [True]
    assert torch.all(result["text_embeddings"][0] == 4)

def test_text_cycle_aligns_to_sensor_label_cycle():
    sensor_record = {
        "unit_id": 12,
        "cycle": 45,  # sourced from sensor label_cycle
        "split": "validation",
        "sensor_embedding": [1.0] * 64,
        "rul": 100.0,
    }

    text_record = {
        "log_id": "L001",
        "unit_id": 12,
        "cycle": 45,
        "split": "validation",
        "text_embedding": [2.0] * 768,
    }

    result = build_fusion_samples(
        [sensor_record],
        [text_record],
    )

    assert result["unit_id"].tolist() == [12]
    assert result["cycle"].tolist() == [45]
    assert result["has_log"].tolist() == [True]
    assert torch.all(result["text_embeddings"][0] == 2)
