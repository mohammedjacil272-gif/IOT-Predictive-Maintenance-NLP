from pathlib import Path

from scripts.verify_fd001_artifact_handover import verify_artifacts


def test_fd001_handover_hashes_and_sample_inference_are_valid():
    result = verify_artifacts(Path(__file__).parents[1])

    assert result["status"] == "PASS"
    assert result["model_input"] == [30, 17]
    assert result["sample_prediction_finite"] is True
    assert result["official_test_evaluation"] == "not_run"
    assert len(result["hashes"]) == 6
