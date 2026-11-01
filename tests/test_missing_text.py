import torch

from training.fusion_dataset import build_fusion_samples as _build_fusion_samples


def build_fusion_samples(sensor_records, text_records, **kwargs):
    return _build_fusion_samples(
        [dict(row, namespace=row.get("namespace", "synthetic_fixture")) for row in sensor_records],
        [dict(row, namespace=row.get("namespace", "synthetic_fixture")) for row in text_records],
        missing_log_policy=kwargs.pop("missing_log_policy", "zero_vector"),
        multiple_log_policy=kwargs.pop("multiple_log_policy", "mean_sorted_log_id"),
        **kwargs,
    )


def test_missing_text_keeps_sensor_window():
    sensor_records = [
        {
            "unit_id": 1,
            "cycle": 30,
            "split": "train",
            "sensor_embedding": [0.1] * 64,
            "rul": 100.0,
        }
    ]

    result = build_fusion_samples(sensor_records, [])

    assert len(result["unit_id"]) == 1
    assert result["text_embeddings"].shape == (1, 768)
    assert torch.count_nonzero(result["text_embeddings"]) == 0
    assert result["has_log"].tolist() == [False]
    assert result["rul"].tolist() == [100.0]


def test_present_text_is_marked():
    sensor_records = [
        {
            "unit_id": 1,
            "cycle": 30,
            "split": "train",
            "sensor_embedding": [0.1] * 64,
            "rul": 100.0,
        }
    ]
    text_records = [
        {
            "log_id": "log-1",
            "unit_id": 1,
            "cycle": 30,
            "split": "train",
            "text_embedding": [0.2] * 768,
        }
    ]

    result = build_fusion_samples(sensor_records, text_records)

    assert result["has_log"].tolist() == [True]
    assert torch.allclose(
        result["text_embeddings"],
        torch.full((1, 768), 0.2),
    )
