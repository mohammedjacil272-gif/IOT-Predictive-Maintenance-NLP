from pathlib import Path

import pytest

from scripts.verify_lstm_baseline import (
    verify_data_artifacts,
    verify_trained_artifacts,
)


def test_missing_baseline_checkpoints_fail_without_regeneration():
    root = Path(__file__).resolve().parent / "missing-baseline-fixture"
    data_dir = root / "data"
    model_dir = root / "models"
    metrics_dir = root / "metrics"
    embeddings_dir = root / "embeddings"
    with pytest.raises(FileNotFoundError, match="read-only.*not regenerate"):
        verify_trained_artifacts(data_dir, model_dir, metrics_dir, embeddings_dir)
    assert not root.exists()


def test_missing_data_artifacts_fail_without_creation():
    data_dir = Path(__file__).resolve().parent / "missing-data-fixture"
    with pytest.raises(FileNotFoundError, match="Missing baseline data artifacts"):
        verify_data_artifacts(data_dir)
    assert not data_dir.exists()
