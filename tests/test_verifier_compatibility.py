from pathlib import Path

from scripts.verify_nlp_fd001_compatibility import build_fixture_samples
from training.nlp_artifact_validation import _artifact_path, validate_nlp_artifact_package


def test_fd001_compatibility_wrapper_uses_its_declared_missing_log_policy():
    sensor = [{
        "unit_id": 1,
        "cycle": 30,
        "split": "fixture",
        "sensor_embedding": [0.25] * 64,
        "rul": 5.0,
    }]

    samples = build_fixture_samples(sensor, [])

    assert samples["sensor_embeddings"].shape == (1, 64)
    assert samples["text_embeddings"].shape == (1, 768)
    assert samples["has_log"].tolist() == [False]


def test_nlp_validator_resolves_current_namespaced_fixture_and_validates_manifest():
    repository_root = Path(__file__).resolve().parents[1]
    artifact_root = repository_root / "artifacts"
    assert _artifact_path(artifact_root, "processed_logs.csv") == (
        artifact_root / "data" / "fd001_nlp_fixture" / "processed_logs.csv"
    )
    assert _artifact_path(artifact_root, "embeddings.pt") == (
        artifact_root / "embeddings" / "fd001_nlp_fixture" / "embeddings.pt"
    )

    report = validate_nlp_artifact_package(artifact_root)

    assert report["status"] == "PASS"
    assert report["genuine_data_claim"] is False
    required = next(check for check in report["checks"] if check["name"] == "required_files")
    assert required["status"] == "PASS"
