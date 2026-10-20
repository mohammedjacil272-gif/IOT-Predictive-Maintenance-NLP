import subprocess
import sys
import tempfile
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


class FailingScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        raise RuntimeError("scaler unavailable")


class InvalidOutputModel:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def eval(self) -> "InvalidOutputModel":
        return self

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return self.output


class FailingModel:
    def eval(self) -> "FailingModel":
        return self

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("model unavailable")


class TestLSTMInference(unittest.TestCase):
    def test_final_tensor_preserves_exact_17_feature_order(self) -> None:
        scaler = IdentityScaler()
        tensor, _, _, _ = build_model_input(records(), scaler)

        expected = np.arange(1, len(SELECTED_FEATURES) + 1, dtype=np.float32)
        np.testing.assert_array_equal(tensor[0, 0].numpy(), expected)
        np.testing.assert_array_equal(scaler.values[0], expected)

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

    def test_cycle_boundaries_and_latest_contiguous_window(self) -> None:
        with self.assertRaisesRegex(LSTMInferenceError, "At least one"):
            build_model_input(records(1, 0), IdentityScaler())

        for count in (1, 29):
            with self.subTest(count=count), self.assertRaisesRegex(LSTMInferenceError, "At least 30"):
                build_model_input(records(1, count), IdentityScaler())

        _, unit_id, start, end = build_model_input(records(1, 30), IdentityScaler())
        self.assertEqual((unit_id, start, end), (7, 1, 30))

        _, unit_id, start, end = build_model_input(records(1, 31), IdentityScaler())
        self.assertEqual((unit_id, start, end), (7, 2, 31))

        with self.assertRaisesRegex(LSTMInferenceError, "Cycle gap"):
            build_model_input(
                records(1, 14) + records(16, 17),
                IdentityScaler(),
            )

    def test_latest_window_values_are_selected_after_cycle_sorting(self) -> None:
        source = records(1, 35)
        for row in source:
            row["setting_1"] = float(row["cycle"])

        tensor, _, start, end = build_model_input(list(reversed(source)), IdentityScaler())

        self.assertEqual((start, end), (6, 35))
        self.assertEqual(float(tensor[0, 0, 0]), 6.0)
        self.assertEqual(float(tensor[0, -1, 0]), 35.0)

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

    def test_malformed_queried_records_are_rejected(self) -> None:
        cases = (
            (None, "must be a mapping"),
            ({"unit_id": "7", "cycle": 1}, "missing required fields"),
            ({**records(1, 1)[0], "unexpected": 1}, "unexpected fields"),
        )
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(LSTMInferenceError, message):
                normalize_sensor_records([value])  # type: ignore[list-item]

    def test_duplicate_cycles_and_mixed_units_are_rejected(self) -> None:
        duplicate = records(1, 30)
        duplicate[1]["cycle"] = duplicate[0]["cycle"]
        with self.assertRaisesRegex(LSTMInferenceError, "Duplicate cycle"):
            normalize_sensor_records(duplicate)

        mixed = records()
        mixed[-1]["unit_id"] = "8"
        with self.assertRaisesRegex(LSTMInferenceError, "exactly one unit_id"):
            normalize_sensor_records(mixed)

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

    def test_scaler_failure_is_reported_as_inference_error(self) -> None:
        with self.assertRaisesRegex(LSTMInferenceError, "Frozen scaler failed"):
            build_model_input(records(), FailingScaler())

    def test_model_failure_is_reported_as_inference_error(self) -> None:
        with self.assertRaisesRegex(LSTMInferenceError, "Frozen model failed"):
            predict_records(records(), scaler=IdentityScaler(), model=FailingModel())

    def test_invalid_model_outputs_are_rejected(self) -> None:
        for output in (torch.empty(0), torch.tensor([1.0, 2.0]), torch.tensor([float("nan")])):
            with self.subTest(output=output), self.assertRaisesRegex(LSTMInferenceError, "invalid prediction"):
                predict_records(
                    records(),
                    scaler=IdentityScaler(),
                    model=InvalidOutputModel(output),
                )

    def test_cli_reports_malformed_json_with_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "records.json"
            input_path.write_text("{not-json", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "iot.lstm_inference", str(input_path)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("LSTM inference error:", result.stdout)

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
