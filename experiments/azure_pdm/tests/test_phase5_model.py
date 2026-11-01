from __future__ import annotations

import pytest
import torch

from experiments.azure_pdm.model import AzureTemporalClassifier


def test_azure_classifier_shape_and_six_class_head():
    model = AzureTemporalClassifier(telemetry_channels=4, metadata_dim=6, class_count=6)
    output = model(torch.zeros((3, 24, 4)), torch.zeros((3, 6)))
    assert output.shape == (3, 6)


def test_azure_classifier_rejects_wrong_metadata_shape():
    model = AzureTemporalClassifier(telemetry_channels=4, metadata_dim=6, class_count=6)
    with pytest.raises(ValueError, match="metadata"):
        model(torch.zeros((3, 3, 4)), torch.zeros((3, 5)))


def test_azure_classifier_rejects_non_finite_input():
    model = AzureTemporalClassifier(telemetry_channels=4, metadata_dim=6, class_count=6)
    telemetry = torch.zeros((1, 3, 4))
    telemetry[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(telemetry, torch.zeros((1, 6)))
