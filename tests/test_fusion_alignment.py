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
