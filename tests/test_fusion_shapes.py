import pytest
import torch

from models.multimodal_fusion import MultimodalFusionModel


def test_output_shape():
    model = MultimodalFusionModel(use_has_log=False)
    sensor = torch.randn(8, 64)
    text = torch.randn(8, 768)

    predictions = model(sensor, text)

    assert predictions.shape == (8,)
    assert torch.isfinite(predictions).all()


def test_single_sample():
    model = MultimodalFusionModel(use_has_log=False)
    predictions = model(torch.randn(1, 64), torch.randn(1, 768))

    assert predictions.shape == (1,)


def test_mismatched_batch_sizes():
    model = MultimodalFusionModel(use_has_log=False)

    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.randn(4, 64), torch.randn(3, 768))


def test_wrong_embedding_dimension():
    model = MultimodalFusionModel(use_has_log=False)

    with pytest.raises(ValueError, match="sensor features"):
        model(torch.randn(4, 63), torch.randn(4, 768))


def test_non_finite_input():
    model = MultimodalFusionModel(use_has_log=False)
    sensor = torch.randn(4, 64)
    text = torch.randn(4, 768)
    text[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        model(sensor, text)


def test_configurable_mlp_dimensions_and_deterministic_eval():
    model = MultimodalFusionModel(
        sensor_dim=4,
        text_dim=6,
        hidden_dim=9,
        head_dim=5,
        dropout=0.5,
        use_has_log=False,
    )
    model.eval()
    sensor = torch.randn(3, 4)
    text = torch.randn(3, 6)

    first = model(sensor, text)
    second = model(sensor, text)

    assert first.shape == (3,)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_has_log_feature_selection_is_explicit():
    with pytest.raises(ValueError, match="Select use_has_log explicitly"):
        MultimodalFusionModel()

    model = MultimodalFusionModel(use_has_log=True)
    sensor = torch.randn(2, 64)
    text = torch.randn(2, 768)
    with pytest.raises(ValueError, match="has_log must have shape"):
        model(sensor, text)
    result = model(sensor, text, has_log=torch.tensor([True, False]))
    assert result.shape == (2,)
    assert torch.isfinite(result).all()
