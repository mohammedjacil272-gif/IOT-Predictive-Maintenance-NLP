from __future__ import annotations

import pytest
import torch

from experiments.azure_pdm.architectures import AzureAggregateMLPClassifier, AzureGRUClassifier, AzureLSTMClassifier


@pytest.mark.parametrize("factory", [AzureLSTMClassifier, AzureGRUClassifier, AzureAggregateMLPClassifier])
def test_alternative_architectures_have_six_class_output(factory):
    model = factory(metadata_dim=6)
    assert model(torch.zeros((2, 24, 4)), torch.zeros((2, 6))).shape == (2, 6)
