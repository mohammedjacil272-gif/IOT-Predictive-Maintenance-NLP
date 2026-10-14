import pytest
import torch

from models.multimodal_fusion import MultimodalFusionModel


def test_output_shape():
    model = MultimodalFusionModel()
    sensor = torch.randn(8, 64)
    text = torch.randn(8, 768)

    predictions = model(sensor, text)

    assert predictions.shape == (8,)
    assert torch.isfinite(predictions).all()


def test_single_sample():
    model = MultimodalFusionModel()
    predictions = model(torch.randn(1, 64), torch.randn(1, 768))

    assert predictions.shape == (1,)


def test_mismatched_batch_sizes():
    model = MultimodalFusionModel()

    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.randn(4, 64), torch.randn(3, 768))


def test_wrong_embedding_dimension():
    model = MultimodalFusionModel()

    with pytest.raises(ValueError, match="sensor features"):
        model(torch.randn(4, 63), torch.randn(4, 768))


def test_non_finite_input():
    model = MultimodalFusionModel()
    sensor = torch.randn(4, 64)
    text = torch.randn(4, 768)
    text[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        model(sensor, text)
