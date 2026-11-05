"""Standalone Azure PdM inference interface for the selected benchmark model."""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .feature_engineering import build_feature_matrix
from .train import LABELS, TELEMETRY_FIELDS


class AzureInferenceError(ValueError):
    """Raised when an input window or model artifact is invalid."""


class AzurePdMPredictor:
    def __init__(self, artifact_path: str | Path) -> None:
        path = Path(artifact_path)
        try:
            with path.open("rb") as stream:
                artifact = pickle.load(stream)
        except Exception as exc:  # pragma: no cover - exact pickle errors vary
            raise AzureInferenceError(f"unable to load model artifact: {exc}") from exc
        required = {"classifier", "lookback_hours", "feature_schema", "preprocessing", "provenance"}
        missing = required.difference(artifact)
        if missing:
            raise AzureInferenceError(f"model artifact is missing fields: {sorted(missing)}")
        self.artifact = artifact

    def _row_features(self, row: dict[str, Any]) -> tuple[datetime, np.ndarray]:
        required = {"datetime", *TELEMETRY_FIELDS}
        if set(row) != required:
            raise AzureInferenceError(f"telemetry row must contain exactly {sorted(required)}")
        try:
            timestamp = datetime.fromisoformat(str(row["datetime"]))
            values = np.asarray([float(row[field]) for field in TELEMETRY_FIELDS], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise AzureInferenceError("telemetry timestamp or numeric value is malformed") from exc
        if timestamp.tzinfo is not None:
            raise AzureInferenceError("timezone-aware timestamps are not accepted; preserve source-naive timestamps")
        if not np.isfinite(values).all():
            raise AzureInferenceError("telemetry values must be finite")
        return timestamp, values

    def _vector(self, payload: dict[str, Any]) -> np.ndarray:
        if set(payload) != {"machine_id", "machine_model", "machine_age", "telemetry"}:
            raise AzureInferenceError("payload must contain machine_id, machine_model, machine_age, and telemetry")
        try:
            machine_id = int(payload["machine_id"])
            age = float(payload["machine_age"])
            model = str(payload["machine_model"])
        except (TypeError, ValueError) as exc:
            raise AzureInferenceError("machine metadata is malformed") from exc
        if machine_id <= 0 or not model or not np.isfinite(age):
            raise AzureInferenceError("machine metadata is invalid")
        rows = payload["telemetry"]
        if not isinstance(rows, list) or len(rows) != int(self.artifact["lookback_hours"]):
            raise AzureInferenceError(f"exactly {self.artifact['lookback_hours']} hourly observations are required")
        parsed = [self._row_features(row) for row in rows]
        parsed.sort(key=lambda item: item[0])
        timestamps = [item[0] for item in parsed]
        expected = [timestamps[0] + timedelta(hours=i) for i in range(len(timestamps))]
        if timestamps != expected:
            raise AzureInferenceError("telemetry window must contain contiguous hourly timestamps")
        telemetry = np.asarray([item[1] for item in parsed], dtype=np.float32)[None, :, :]
        prep = self.artifact["preprocessing"]
        mean = np.asarray(prep["telemetry_mean"], dtype=np.float32)
        std = np.asarray(prep["telemetry_std"], dtype=np.float32)
        telemetry = (telemetry - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        metadata = np.zeros((1, len(prep["model_vocab"]) + 1), dtype=np.float32)
        metadata[0, 0] = (age - float(prep["age_mean"])) / float(prep["age_std"])
        metadata[0, prep["model_vocab"].get(model, 0)] = 1.0
        return build_feature_matrix(telemetry, metadata, engineered=bool(self.artifact["engineered"]), include_metadata=bool(self.artifact.get("include_metadata", True)))

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        vector = self._vector(payload)
        probabilities = self.artifact["classifier"].predict_proba(vector)[0].astype(float)
        thresholds = self.artifact.get("thresholds")
        if thresholds is None:
            prediction = int(np.argmax(probabilities))
        else:
            threshold_array = np.asarray(thresholds, dtype=np.float32)
            eligible = probabilities >= threshold_array
            if eligible.any():
                adjusted = np.where(eligible, probabilities / threshold_array, -np.inf)
                prediction = int(np.argmax(adjusted))
            else:
                prediction = int(np.argmax(probabilities))
        return {"machine_id": int(payload["machine_id"]), "predicted_class": LABELS[prediction], "probabilities": {label: float(probabilities[i]) for i, label in enumerate(LABELS)}, "model_version": self.artifact["provenance"]["model_version"], "provenance": self.artifact["provenance"]}

    def predict_batch(self, payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_one(payload) for payload in payloads]
