
import pytest
import torch

from nlp.fusion import prepare_fusion


def make_embedding(value):
    return torch.full((768,), float(value))


def test_multiple_logs_are_averaged_deterministically():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"}
    ]
    logs = [
        {"log_id": "L002", "unit_id": 1, "cycle": 10, "source_split": "train"},
        {"log_id": "L001", "unit_id": 1, "cycle": 10, "source_split": "train"},
    ]
    embeddings = torch.stack([make_embedding(4), make_embedding(2)])

    result = prepare_fusion(sensors, logs, embeddings)

    assert result["metadata"][0]["has_log"] is True
    assert result["metadata"][0]["log_ids"] == ["L001", "L002"]
    assert torch.allclose(
        result["embeddings"][0], make_embedding(3)
    )


def test_sensor_window_without_log_gets_zero_vector():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"}
    ]

    result = prepare_fusion(
        sensors, [], torch.empty((0, 768))
    )

    assert result["metadata"][0]["has_log"] is False
    assert torch.equal(result["embeddings"][0], torch.zeros(768))


def test_unmatched_log_is_reported():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"}
    ]
    logs = [
        {"log_id": "L999", "unit_id": 2, "cycle": 20, "source_split": "train"}
    ]

    result = prepare_fusion(sensors, logs, make_embedding(1).unsqueeze(0))

    assert result["unmatched_log_ids"] == ["L999"]
    assert result["metadata"][0]["has_log"] is False


def test_split_mismatch_is_rejected():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"}
    ]
    logs = [
        {"log_id": "L001", "unit_id": 1, "cycle": 10, "source_split": "test"}
    ]

    with pytest.raises(ValueError, match="source_split mismatch"):
        prepare_fusion(sensors, logs, make_embedding(1).unsqueeze(0))


def test_duplicate_log_id_is_rejected():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"}
    ]
    logs = [
        {"log_id": "L001", "unit_id": 1, "cycle": 10, "source_split": "train"},
        {"log_id": "L001", "unit_id": 1, "cycle": 10, "source_split": "train"},
    ]
    embeddings = torch.stack([make_embedding(1), make_embedding(2)])

    with pytest.raises(ValueError, match="Duplicate log_id"):
        prepare_fusion(sensors, logs, embeddings)

def test_unalignable_log_is_reported():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"}
    ]
    logs = [
        {"log_id": "L100", "unit_id": None, "cycle": 10, "source_split": "train"}
    ]

    result = prepare_fusion(sensors, logs, make_embedding(1).unsqueeze(0))

    assert result["unalignable_log_ids"] == ["L100"]
    assert result["metadata"][0]["has_log"] is False


def test_invalid_embedding_dimensions_are_rejected():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"}
    ]
    logs = [
        {"log_id": "L001", "unit_id": 1, "cycle": 10, "source_split": "train"}
    ]

    with pytest.raises(ValueError, match="shape"):
        prepare_fusion(sensors, logs, torch.zeros((1, 512)))


def test_duplicate_sensor_window_key_is_rejected():
    sensors = [
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"},
        {"unit_id": 1, "label_cycle": 10, "source_split": "train"},
    ]

    with pytest.raises(ValueError, match="Duplicate sensor window key"):
        prepare_fusion(sensors, [], torch.empty((0, 768)))
