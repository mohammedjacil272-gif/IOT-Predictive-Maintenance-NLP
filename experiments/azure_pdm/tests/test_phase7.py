import pickle
from datetime import datetime, timedelta

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.azure_pdm.feature_engineering import build_feature_matrix, feature_names
from experiments.azure_pdm.inference import AzureInferenceError, AzurePdMPredictor


def _artifact(tmp_path):
    telemetry = np.random.default_rng(4).normal(size=(36, 3, 4)).astype(np.float32)
    metadata = np.zeros((36, 3), dtype=np.float32)
    x = build_feature_matrix(telemetry, metadata, engineered=False, include_metadata=True)
    y = np.arange(36) % 6
    classifier = LogisticRegression(max_iter=100).fit(x, y)
    artifact = {
        "classifier": classifier,
        "lookback_hours": 3,
        "engineered": False,
        "include_metadata": True,
        "feature_schema": feature_names(engineered=False, include_metadata=True),
        "preprocessing": {"telemetry_mean": [0, 0, 0, 0], "telemetry_std": [1, 1, 1, 1], "age_mean": 0, "age_std": 1, "model_vocab": {"<UNK>": 0, "A": 1}},
        "provenance": {"model_version": "test", "contract_version": "test", "dataset_sha256": "x", "source_hashes": {}},
    }
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(artifact))
    return path


def _payload():
    start = datetime(2020, 1, 1)
    rows = []
    for i in range(3):
        rows.append({"datetime": (start + timedelta(hours=i)).isoformat(), "volt": 1.0, "rotate": 2.0, "pressure": 3.0, "vibration": 4.0})
    return {"machine_id": 1, "machine_model": "A", "machine_age": 5.0, "telemetry": rows}


def test_feature_matrix_is_finite_and_schema_is_stable():
    telemetry = np.ones((2, 3, 4), dtype=np.float32)
    metadata = np.zeros((2, 2), dtype=np.float32)
    matrix = build_feature_matrix(telemetry, metadata, engineered=True, include_metadata=True)
    assert matrix.shape[1] == len(feature_names(engineered=True, include_metadata=True))
    assert np.isfinite(matrix).all()


def test_inference_is_deterministic_and_returns_six_class_contract_fields(tmp_path):
    predictor = AzurePdMPredictor(_artifact(tmp_path))
    first = predictor.predict_one(_payload())
    second = predictor.predict_batch([_payload()])[0]
    assert first == second
    assert set(first["probabilities"]) == {"none", "comp1", "comp2", "comp3", "comp4", "multi_failure"}
    assert first["model_version"] == "test"


def test_inference_rejects_gap_and_wrong_row_shape(tmp_path):
    predictor = AzurePdMPredictor(_artifact(tmp_path))
    payload = _payload()
    payload["telemetry"][1]["datetime"] = datetime(2020, 1, 1, 3).isoformat()
    with pytest.raises(AzureInferenceError):
        predictor.predict_one(payload)
    payload = _payload()
    payload["telemetry"][0]["volt"] = "bad"
    with pytest.raises(AzureInferenceError):
        predictor.predict_one(payload)
