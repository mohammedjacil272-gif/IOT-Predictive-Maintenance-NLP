"""Prepare InfluxDB sensor records for the frozen FD001 LSTM baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from numbers import Real
from typing import Any, Iterable, Mapping, Protocol, Sequence

import joblib
import numpy as np
import torch

from models.lstm_baseline import LSTMBaseline
from preprocessing.cmapss_fd001 import SELECTED_FEATURES


WINDOW_SIZE = 30
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALER_PATH = PROJECT_ROOT / "artifacts/data/fd001_lstm_baseline/scaler.joblib"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts/models/fd001_lstm_baseline/best_model.pt"

INFLUX_METADATA_FIELDS = frozenset({"_measurement", "_time", "_start", "_stop", "result", "table"})
FORBIDDEN_LABEL_FIELDS = frozenset({"rul", "RUL", "failure_label", "failure", "label"})
ALLOWED_FIELDS = frozenset({"unit_id", "cycle", *SELECTED_FEATURES})


class LSTMInferenceError(ValueError):
    """Raised when records cannot form a valid baseline inference window."""


class Scaler(Protocol):
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform feature rows using an already-fitted scaler."""


class Model(Protocol):
    def eval(self) -> Any:
        """Set the model to evaluation mode."""

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Predict from a model-ready input tensor."""


@dataclass(frozen=True)
class InferenceResult:
    """Prediction and identity of the input window."""

    unit_id: int
    window_start_cycle: int
    window_end_cycle: int
    prediction: float
    input_tensor: torch.Tensor


def normalize_sensor_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, int | float]]:
    """Validate and normalize pivoted InfluxDB records.

    Influx metadata is discarded. The returned dictionaries contain only the
    identifiers and the exact 17 baseline features, in baseline feature order.
    """

    normalized: list[dict[str, int | float]] = []
    unit_ids: set[int] = set()
    seen_cycles: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise LSTMInferenceError(f"Record {index} must be a mapping.")
        forbidden = sorted(FORBIDDEN_LABEL_FIELDS.intersection(record.keys()))
        if forbidden:
            raise LSTMInferenceError(f"Record {index} contains forbidden label fields: {forbidden}")
        unexpected = sorted(set(record) - ALLOWED_FIELDS - INFLUX_METADATA_FIELDS)
        if unexpected:
            raise LSTMInferenceError(f"Record {index} contains unexpected fields: {unexpected}")
        missing = sorted(ALLOWED_FIELDS - set(record))
        if missing:
            raise LSTMInferenceError(f"Record {index} is missing required fields: {missing}")

        unit_id = _positive_integer(record["unit_id"], "unit_id", index, allow_text=True)
        cycle = _positive_integer(record["cycle"], "cycle", index)
        if cycle in seen_cycles:
            raise LSTMInferenceError(f"Duplicate cycle {cycle} found for unit_id {unit_id}.")
        seen_cycles.add(cycle)
        unit_ids.add(unit_id)
        row: dict[str, int | float] = {"unit_id": unit_id, "cycle": cycle}
        for feature in SELECTED_FEATURES:
            value = record[feature]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise LSTMInferenceError(f"Record {index} feature {feature} must be numeric.")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise LSTMInferenceError(f"Record {index} feature {feature} must be finite.")
            row[feature] = numeric
        normalized.append(row)

    if not normalized:
        raise LSTMInferenceError("At least one sensor record is required.")
    if len(unit_ids) != 1:
        raise LSTMInferenceError(f"Records must contain exactly one unit_id; found {sorted(unit_ids)}.")
    return sorted(normalized, key=lambda row: int(row["cycle"]))


def build_model_input(records: Iterable[Mapping[str, object]], scaler: Scaler) -> tuple[torch.Tensor, int, int, int]:
    """Build one scaled chronological ``(1, 30, 17)`` input tensor."""

    normalized = normalize_sensor_records(records)
    if len(normalized) < WINDOW_SIZE:
        raise LSTMInferenceError(
            f"At least {WINDOW_SIZE} records are required; received {len(normalized)}."
        )
    cycles = np.asarray([int(row["cycle"]) for row in normalized], dtype=np.int64)
    expected = np.arange(cycles[0], cycles[-1] + 1, dtype=np.int64)
    if not np.array_equal(cycles, expected):
        raise LSTMInferenceError(
            f"Cycle gap detected for unit_id {normalized[0]['unit_id']}: "
            f"expected consecutive cycles from {cycles[0]} to {cycles[-1]}.")

    window = normalized[-WINDOW_SIZE:]
    values = np.asarray(
        [[float(row[feature]) for feature in SELECTED_FEATURES] for row in window],
        dtype=np.float64,
    )
    try:
        scaled = np.asarray(scaler.transform(values), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - provide an adapter-specific error.
        raise LSTMInferenceError(f"Frozen scaler failed to transform sensor records: {exc}") from exc
    if scaled.shape != (WINDOW_SIZE, len(SELECTED_FEATURES)) or not np.isfinite(scaled).all():
        raise LSTMInferenceError(f"Scaler returned invalid shape or values: {scaled.shape}")
    tensor = torch.from_numpy(scaled).unsqueeze(0).to(dtype=torch.float32)
    return tensor, int(window[0]["unit_id"]), int(window[0]["cycle"]), int(window[-1]["cycle"])


def load_frozen_model(model_path: Path = DEFAULT_MODEL_PATH) -> LSTMBaseline:
    """Load and validate the existing frozen baseline checkpoint."""

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    if config.get("input_size") != len(SELECTED_FEATURES) or config.get("sequence_length") != WINDOW_SIZE:
        raise LSTMInferenceError("Checkpoint configuration does not match the frozen 30-by-17 baseline.")
    model = LSTMBaseline(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        lstm_dropout=float(config["lstm_dropout"]),
        head_dropout=float(config["head_dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_records(
    records: Sequence[Mapping[str, object]],
    *,
    scaler_path: Path = DEFAULT_SCALER_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler: Scaler | None = None,
    model: Model | None = None,
) -> InferenceResult:
    """Scale records with the frozen scaler and predict RUL with the frozen model."""

    fitted_scaler = scaler if scaler is not None else joblib.load(scaler_path)
    input_tensor, unit_id, start_cycle, end_cycle = build_model_input(records, fitted_scaler)
    inference_model = model if model is not None else load_frozen_model(model_path)
    try:
        inference_model.eval()
        with torch.no_grad():
            output = inference_model(input_tensor)
        prediction_array = output.detach().cpu().reshape(-1).numpy()
    except Exception as exc:  # noqa: BLE001 - normalize model failures at the adapter boundary.
        raise LSTMInferenceError(f"Frozen model failed during inference: {exc}") from exc
    if prediction_array.size != 1 or not np.isfinite(prediction_array[0]):
        raise LSTMInferenceError("Model returned an invalid prediction.")
    return InferenceResult(unit_id, start_cycle, end_cycle, float(prediction_array[0]), input_tensor)


def _positive_integer(value: object, field: str, index: int, *, allow_text: bool = False) -> int:
    if isinstance(value, bool):
        raise LSTMInferenceError(f"Record {index} {field} must be a positive integer.")
    if isinstance(value, str):
        if not allow_text or not value.isdecimal():
            raise LSTMInferenceError(f"Record {index} {field} must be a positive integer.")
        value = int(value)
    if not isinstance(value, Real):
        raise LSTMInferenceError(f"Record {index} {field} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LSTMInferenceError(f"Record {index} {field} must be a positive integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise LSTMInferenceError(f"Record {index} {field} must be a positive integer.")
    return int(numeric)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen FD001 LSTM inference on queried JSON records.")
    parser.add_argument("records_json", help="JSON file containing a list of pivoted InfluxDB records, or - for stdin")
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        text = sys.stdin.read() if args.records_json == "-" else Path(args.records_json).read_text(encoding="utf-8")
        result = predict_records(
            json.loads(text), scaler_path=args.scaler_path, model_path=args.model_path
        )
        print(json.dumps({
            "unit_id": result.unit_id,
            "window_start_cycle": result.window_start_cycle,
            "window_end_cycle": result.window_end_cycle,
            "prediction": result.prediction,
        }, sort_keys=True))
    except (LSTMInferenceError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"LSTM inference error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
