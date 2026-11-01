"""Bounded Phase 7 feature, imbalance, calibration, and inference-artifact run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import ExtraTreesClassifier

from .feature_engineering import build_feature_matrix, feature_names
from .inference import AzurePdMPredictor
from .train import LABELS, load_machines, load_rows, load_telemetry, metrics, prepare_arrays, sha256_file


def ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1); predictions = probabilities.argmax(axis=1)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(float((predictions[mask] == y_true[mask]).mean()) - float(confidence[mask].mean()))
    return result


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, *, class_weight: str | None = "balanced") -> LogisticRegression:
    model = LogisticRegression(max_iter=200, class_weight=class_weight, solver="lbfgs", random_state=42)
    model.fit(train_x, train_y)
    return model


def evaluate(model: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    probabilities = model.predict_proba(x)
    result = metrics(y, probabilities)
    result["ece"] = ece(y, probabilities)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--temporal-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    telemetry = load_telemetry(args.dataset_root / "PdM_telemetry.csv")
    machines = load_machines(args.dataset_root / "PdM_machines.csv")
    all_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_arrays: dict[str, Any] | None = None
    selected_preprocessing: dict[str, Any] | None = None
    selected_spec: dict[str, Any] | None = None
    for lookback in (3, 24):
        rows = load_rows(args.temporal_dataset, lookback)
        arrays, preprocessing = prepare_arrays(rows, telemetry, machines, seed=42)
        for engineered in (False, True):
            for include_metadata in (False, True):
                x = build_feature_matrix(arrays["telemetry"], arrays["metadata"], engineered=engineered, include_metadata=include_metadata)
                train, val, test = arrays["train_indices"], arrays["validation_indices"], arrays["test_indices"]
                model = fit_logistic(x[train], arrays["labels"][train])
                val_result = evaluate(model, x[val], arrays["labels"][val])
                record = {"lookback_hours": lookback, "engineered": engineered, "include_metadata": include_metadata, "feature_names": feature_names(engineered=engineered, include_metadata=include_metadata), "validation": val_result}
                all_results.append(record)
                if selected is None or val_result["macro_f1"] > selected["validation"]["macro_f1"]:
                    selected = record
                    selected_arrays = arrays
                    selected_preprocessing = preprocessing
                    selected_spec = {"rows": rows, "x": x, "model": model, "lookback": lookback, "engineered": engineered, "include_metadata": include_metadata}
        # Optional installed tree comparison on the engineered metadata representation, validation only.
        tree_x = build_feature_matrix(arrays["telemetry"], arrays["metadata"], engineered=True, include_metadata=True)
        train, val = arrays["train_indices"], arrays["validation_indices"]
        tree = ExtraTreesClassifier(n_estimators=80, max_depth=18, class_weight="balanced", random_state=42, n_jobs=4)
        tree.fit(tree_x[train], arrays["labels"][train])
        all_results.append({"lookback_hours": lookback, "model": "extra_trees", "validation": evaluate(tree, tree_x[val], arrays["labels"][val])})
    assert selected is not None and selected_arrays is not None and selected_preprocessing is not None and selected_spec is not None
    x = selected_spec["x"]; arrays = selected_arrays; train, val, test = arrays["train_indices"], arrays["validation_indices"], arrays["test_indices"]
    final_model = fit_logistic(x[train], arrays["labels"][train])
    test_result = evaluate(final_model, x[test], arrays["labels"][test])
    val_prob = final_model.predict_proba(x[val])
    test_prob = final_model.predict_proba(x[test])
    model_dir = args.output_root / "selected_logistic"
    model_dir.mkdir()
    prep = {"telemetry_mean": selected_preprocessing["telemetry_mean_fit_on_train"], "telemetry_std": selected_preprocessing["telemetry_std_fit_on_train"], "age_mean": selected_preprocessing["machine_age_mean_fit_on_train"], "age_std": selected_preprocessing["machine_age_std_fit_on_train"], "model_vocab": selected_preprocessing["machine_model_vocab_fit_on_train"]}
    artifact = {"classifier": final_model, "lookback_hours": selected_spec["lookback"], "engineered": selected_spec["engineered"], "include_metadata": selected_spec["include_metadata"], "feature_schema": feature_names(engineered=selected_spec["engineered"], include_metadata=selected_spec["include_metadata"]), "preprocessing": prep, "provenance": {"model_version": "azure_pdm_phase7_logistic_v1", "contract_version": "azure_pdm_experiment_contract_v1.1", "dataset_sha256": sha256_file(args.temporal_dataset), "source_hashes": {name: sha256_file(args.dataset_root / name) for name in ("PdM_telemetry.csv", "PdM_errors.csv", "PdM_maint.csv", "PdM_failures.csv", "PdM_machines.csv")}}}
    model_path = model_dir / "model.pkl"
    with model_path.open("wb") as stream: pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (model_dir / "feature_schema.json").write_text(json.dumps(artifact["feature_schema"], indent=2) + "\n", encoding="utf-8")
    (model_dir / "preprocessing.json").write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (model_dir / "thresholds.json").write_text(json.dumps({"method": "argmax", "thresholds": [1.0] * 6}, indent=2) + "\n", encoding="utf-8")
    model_hash = sha256_file(model_path)
    # Determinism smoke test through the public interface using a real source window.
    first_machine = int(selected_spec["rows"][0]["machineID"]); first_anchor = selected_spec["rows"][0]["anchor_timestamp"]
    from datetime import datetime, timedelta
    anchor = datetime.fromisoformat(first_anchor); window = []
    for offset in range(selected_spec["lookback"]):
        timestamp = anchor - timedelta(hours=selected_spec["lookback"] - offset)
        values = telemetry[first_machine][timestamp]
        window.append({"datetime": timestamp.isoformat(sep=" "), **dict(zip(("volt", "rotate", "pressure", "vibration"), values))})
    payload = {"machine_id": first_machine, "machine_model": machines[first_machine][0], "machine_age": machines[first_machine][1], "telemetry": window}
    predictor = AzurePdMPredictor(model_path); prediction_a = predictor.predict_one(payload); prediction_b = predictor.predict_one(payload)
    (model_dir / "determinism_check.json").write_text(json.dumps({"identical": prediction_a == prediction_b, "prediction": prediction_a}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"phase": "6.1-7.0", "generated_at": time.time(), "dataset_classification": "simulated benchmark", "selected": {"lookback_hours": selected_spec["lookback"], "engineered": selected_spec["engineered"], "include_metadata": selected_spec["include_metadata"], "feature_names": feature_names(engineered=selected_spec["engineered"], include_metadata=True), "validation": selected["validation"], "test": test_result, "model_sha256": model_hash}, "validation_sweep": all_results, "calibration": {"selected_validation_ece": ece(arrays["labels"][val], val_prob), "selected_test_ece": ece(arrays["labels"][test], test_prob), "method": "uncalibrated logistic probabilities; no test calibration"}, "inference": {"artifact": str(model_path), "deterministic": prediction_a == prediction_b, "batch_supported": True}, "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__}, "training_performed": True, "inference_performed": True, "predictive_performance_claim": False}
    (args.output_root / "phase7_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": manifest["selected"], "output_root": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
