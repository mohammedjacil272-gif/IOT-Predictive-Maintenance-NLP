import unittest
from pathlib import Path

import numpy as np
import torch

from iot.lstm_inference import (
    LSTMInferenceError,
    SELECTED_FEATURES,
    WINDOW_SIZE,
    build_model_input,
    normalize_sensor_records,
    predict_records,
)


def records(start: int = 1, count: int = WINDOW_SIZE) -> list[dict[str, object]]:
    result = []
    for cycle in range(start, start + count):
        row = {"unit_id": "7", "cycle": cycle, "_measurement": "fd001_sensor_reading", "_time": "synthetic"}
        row.update({feature: float(index) for index, feature in enumerate(SELECTED_FEATURES, start=1)})
        result.append(row)
    return result


class IdentityScaler:
    def __init__(self) -> None:
        self.values = None

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.values = values
        return values


class ConstantModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.input = None

    def eval(self) -> "ConstantModel":
        self.eval_called = True
        return self

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        self.input = values
        return torch.tensor([12.5], dtype=torch.float32)


class TestLSTMInference(unittest.TestCase):
    def test_normalization_ignores_influx_metadata_and_orders_cycles(self) -> None:
        normalized = normalize_sensor_records(list(reversed(records())))
        self.assertEqual([row["cycle"] for row in normalized], list(range(1, 31)))
        self.assertEqual(list(normalized[0]), ["unit_id", "cycle", *SELECTED_FEATURES])
        self.assertNotIn("_time", normalized[0])

    def test_model_input_uses_latest_30_chronological_cycles(self) -> None:
        scaler = IdentityScaler()
        tensor, unit_id, start, end = build_model_input(records(1, 35), scaler)
        self.assertEqual((1, 30, 17), tuple(tensor.shape))
        self.assertEqual((7, 6, 35), (unit_id, start, end))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_malformed_windows_are_rejected(self) -> None:
        for bad_records, message in (
            (records(1, 29), "At least 30"),
            (records(1, 30)[:10] + [records(1, 30)[9]] + records(1, 30)[10:], "Duplicate cycle"),
            (records(1, 31)[:10] + records(1, 31)[11:], "Cycle gap"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(LSTMInferenceError, message):
                build_model_input(bad_records, IdentityScaler())

    def test_invalid_values_units_and_labels_are_rejected(self) -> None:
        bad = records()
        bad[0]["sensor_2"] = float("nan")
        with self.assertRaisesRegex(LSTMInferenceError, "finite"):
            normalize_sensor_records(bad)

        mixed = records()
        mixed[-1]["unit_id"] = "8"
        with self.assertRaisesRegex(LSTMInferenceError, "exactly one unit_id"):
            normalize_sensor_records(mixed)

        labeled = records()
        labeled[0]["rul"] = 10
        with self.assertRaisesRegex(LSTMInferenceError, "forbidden label"):
            normalize_sensor_records(labeled)

    def test_invalid_unit_ids_are_rejected_as_inference_errors(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf"), "not-an-id", True, 0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(LSTMInferenceError, "unit_id must be a positive integer"):
                invalid = records(1, 1)
                invalid[0]["unit_id"] = value
                normalize_sensor_records(invalid)

    def test_invalid_cycles_are_rejected_as_inference_errors(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf"), "not-a-cycle", True, 0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(LSTMInferenceError, "cycle must be a positive integer"):
                invalid = records(1, 1)
                invalid[0]["cycle"] = value
                normalize_sensor_records(invalid)

    def test_predict_records_uses_eval_model_without_labels(self) -> None:
        model = ConstantModel()
        result = predict_records(records(), scaler=IdentityScaler(), model=model)
        self.assertEqual(result.prediction, 12.5)
        self.assertTrue(model.eval_called)
        self.assertEqual(tuple(model.input.shape), (1, 30, 17))

    def test_frozen_artifacts_predict_without_live_services(self) -> None:
        result = predict_records(
            records(),
            scaler_path=Path("artifacts/data/fd001_lstm_baseline/scaler.joblib"),
            model_path=Path("artifacts/models/fd001_lstm_baseline/best_model.pt"),
        )
        self.assertEqual(tuple(result.input_tensor.shape), (1, 30, 17))
        self.assertTrue(np.isfinite(result.prediction))


if __name__ == "__main__":
    unittest.main()
