from pathlib import Path

from scripts.verify_sensor_embeddings import verify_sensor_artifact


def test_checked_in_sensor_embedding_artifact_is_complete_and_aligned():
    result = verify_sensor_artifact(
        Path(__file__).parents[1] / "artifacts" / "embeddings" / "fd001_lstm_baseline"
    )

    assert result["rows"] == 17731
    assert result["dimension"] == 64
    assert result["index_min"] == 0
    assert result["index_max"] == 17730
    assert result["duplicate_keys"] == 0
    assert result["split_counts"] == {"train": 14207, "validation": 3524}
