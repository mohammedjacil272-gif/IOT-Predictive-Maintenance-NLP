from pathlib import Path

import pytest

from scripts.verify_fd001_artifact_handover import verify_artifacts


def test_fd001_handover_hashes_and_sample_inference_are_valid():
    root = Path(__file__).parents[1]
    checkpoint = root / "artifacts/models/fd001_lstm_baseline/best_model.pt"
    if not checkpoint.is_file():
        with pytest.raises(FileNotFoundError, match="Required handover artifact is missing"):
            verify_artifacts(root)
        return
    result = verify_artifacts(root)

    assert result["status"] == "PASS"
    assert result["model_input"] == [30, 17]
    assert result["sample_prediction_finite"] is True
    assert result["official_test_evaluation"] == "completed_final_evaluation_only"
    assert len(result["hashes"]) == 6
