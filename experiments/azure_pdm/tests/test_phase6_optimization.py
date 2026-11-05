import numpy as np

from experiments.azure_pdm.optimize import (
    class_weights,
    expected_calibration_error,
    machine_split_rows,
    threshold_predictions,
    tune_thresholds,
)


def test_class_weights_are_finite_and_capped():
    labels = np.array([0] * 100 + [1] * 5 + [2] * 2, dtype=np.int64)
    weights = class_weights(labels, "capped20")
    assert np.isfinite(weights).all()
    assert weights.max() <= 20.0


def test_threshold_predictions_falls_back_to_argmax_when_no_threshold_matches():
    probabilities = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
    thresholds = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    assert threshold_predictions(probabilities, thresholds).tolist() == [0]


def test_threshold_tuning_uses_classwise_thresholds_and_returns_score():
    y_true = np.array([0, 1, 2, 1], dtype=np.int64)
    probabilities = np.array(
        [[0.80, 0.10, 0.05, 0.02, 0.02, 0.01], [0.45, 0.50, 0.01, 0.01, 0.02, 0.01],
         [0.10, 0.20, 0.60, 0.03, 0.04, 0.03], [0.40, 0.55, 0.01, 0.01, 0.02, 0.01]],
        dtype=np.float32,
    )
    thresholds, score = tune_thresholds(y_true, probabilities)
    assert len(thresholds) == 6
    assert 0.0 <= score <= 1.0


def test_calibration_error_is_zero_for_perfect_confidence():
    y_true = np.array([0, 1], dtype=np.int64)
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert expected_calibration_error(y_true, probabilities) == 0.0


def test_machine_holdout_split_keeps_machine_ids_disjoint():
    rows = [{"machineID": str(machine), "split": "original"} for machine in [1, 60, 61, 80, 81, 100]]
    split = machine_split_rows(rows)
    by_split = {}
    for row in split:
        by_split.setdefault(row["split"], set()).add(row["machineID"])
    assert by_split == {"train": {"1", "60"}, "validation": {"61", "80"}, "test": {"81", "100"}}
