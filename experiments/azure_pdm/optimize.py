"""Bounded Azure PdM architecture, imbalance, threshold, and holdout experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .architectures import AzureAggregateMLPClassifier, AzureGRUClassifier, AzureLSTMClassifier
from .model import AzureTemporalClassifier
from .train import LABELS, LABEL_TO_ID, load_machines, load_rows, load_telemetry, metrics, prepare_arrays, set_seed, sha256_file


ARCHITECTURES: dict[str, Callable[[int], nn.Module]] = {
    "cnn": lambda metadata_dim: AzureTemporalClassifier(telemetry_channels=4, metadata_dim=metadata_dim, class_count=6),
    "lstm": lambda metadata_dim: AzureLSTMClassifier(metadata_dim=metadata_dim),
    "gru": lambda metadata_dim: AzureGRUClassifier(metadata_dim=metadata_dim),
    "mlp": lambda metadata_dim: AzureAggregateMLPClassifier(metadata_dim=metadata_dim),
}


def class_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float32)
    if mode == "unweighted":
        return np.ones(len(LABELS), dtype=np.float32)
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1.0))
    if mode == "capped20":
        weights = np.minimum(weights, 20.0)
    elif mode != "weighted":
        raise ValueError(f"Unknown weight mode: {mode}")
    return weights.astype(np.float32)


def _predict(model: nn.Module, arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(arrays["telemetry"])[indices]
    meta = torch.from_numpy(arrays["metadata"])[indices]
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            logits = model(x[start:start + batch_size].to(device), meta[start:start + batch_size].to(device))
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output, axis=0)


def train_candidate(
    arrays: dict[str, np.ndarray],
    *,
    architecture: str,
    weight_mode: str,
    seed: int,
    epochs: int,
    batch_size: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    output_dir.mkdir(parents=True, exist_ok=False)
    train_idx, val_idx, test_idx = arrays["train_indices"], arrays["validation_indices"], arrays["test_indices"]
    x = torch.from_numpy(arrays["telemetry"])
    meta = torch.from_numpy(arrays["metadata"])
    y = torch.from_numpy(arrays["labels"])
    model = ARCHITECTURES[architecture](meta.shape[1]).to(device)
    weights = class_weights(arrays["labels"][train_idx], weight_mode)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x[train_idx], meta[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True, num_workers=0)
    best_val = -1.0; best_epoch = 0; history = []
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for bx, bm, by in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx.to(device), bm.to(device))
            loss = loss_fn(logits, by.to(device))
            loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * len(by)
        val_prob = _predict(model, arrays, val_idx, device, batch_size * 2)
        val_metrics = metrics(arrays["labels"][val_idx], val_prob)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_idx), "validation_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]; best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "architecture": architecture, "weight_mode": weight_mode, "metadata_dim": meta.shape[1], "class_count": 6, "seed": seed, "best_epoch": epoch, "validation_macro_f1": best_val, "label_mapping": LABEL_TO_ID}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_prob = _predict(model, arrays, val_idx, device, batch_size * 2)
    test_prob = _predict(model, arrays, test_idx, device, batch_size * 2)
    result = {
        "architecture": architecture,
        "weight_mode": weight_mode,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "validation": metrics(arrays["labels"][val_idx], val_prob),
        "test_standard": metrics(arrays["labels"][test_idx], test_prob),
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": history,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "probabilities.npz", validation=val_prob, test=test_prob, validation_labels=arrays["labels"][val_idx], test_labels=arrays["labels"][test_idx])
    return result


def fit_logistic_validation(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    train, val, test = arrays["train_indices"], arrays["validation_indices"], arrays["test_indices"]
    aggregate = np.concatenate([arrays["telemetry"].mean(axis=1), arrays["telemetry"].std(axis=1), arrays["metadata"]], axis=1)
    classifier = LogisticRegression(max_iter=100, class_weight="balanced", solver="lbfgs", random_state=42)
    classifier.fit(aggregate[train], arrays["labels"][train])
    val_prob = classifier.predict_proba(aggregate[val]); test_prob = classifier.predict_proba(aggregate[test])
    return {"architecture": "balanced_logistic_regression", "validation": metrics(arrays["labels"][val], val_prob), "test_standard": metrics(arrays["labels"][test], test_prob)}


def threshold_predictions(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    eligible = probabilities >= thresholds.reshape(1, -1)
    adjusted = probabilities / thresholds.reshape(1, -1)
    adjusted[~eligible] = -np.inf
    none_eligible = ~eligible.any(axis=1)
    predictions = adjusted.argmax(axis=1)
    predictions[none_eligible] = probabilities[none_eligible].argmax(axis=1)
    return predictions


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[list[float], float]:
    thresholds = np.ones(len(LABELS), dtype=np.float32)
    grid = np.arange(0.10, 1.01, 0.10)
    for class_id in range(len(LABELS)):
        best = f1_score(y_true, threshold_predictions(probabilities, thresholds), average="macro", zero_division=0)
        best_threshold = 1.0
        for threshold in grid:
            candidate = thresholds.copy(); candidate[class_id] = threshold
            score = f1_score(y_true, threshold_predictions(probabilities, candidate), average="macro", zero_division=0)
            if score > best:
                best, best_threshold = score, float(threshold)
        thresholds[class_id] = best_threshold
    return thresholds.tolist(), float(f1_score(y_true, threshold_predictions(probabilities, thresholds), average="macro", zero_division=0))


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1); prediction = probabilities.argmax(axis=1); correct = (prediction == y_true).astype(float)
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def machine_split_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Complete-machine split: 1-60 train, 61-80 validation, 81-100 test."""
    output = []
    for row in rows:
        machine = int(row["machineID"])
        row = dict(row)
        row["split"] = "train" if machine <= 60 else "validation" if machine <= 80 else "test"
        output.append(row)
    return output


def error_analysis(rows: list[dict[str, str]], test_indices: np.ndarray, y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    by_machine = defaultdict(lambda: {"count": 0, "errors": 0})
    by_month = defaultdict(lambda: {"count": 0, "errors": 0})
    for index, actual, predicted in zip(test_indices, y_true, predictions):
        row = rows[int(index)]; machine = row["machineID"]; month = row["anchor_timestamp"][:7]
        by_machine[machine]["count"] += 1; by_machine[machine]["errors"] += int(actual != predicted)
        by_month[month]["count"] += 1; by_month[month]["errors"] += int(actual != predicted)
    return {"by_machine": dict(by_machine), "by_month": dict(by_month)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--temporal-dataset", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 6 output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    telemetry = load_telemetry(args.dataset_root / "PdM_telemetry.csv")
    machines = load_machines(args.dataset_root / "PdM_machines.csv")
    sweep: list[dict[str, Any]] = []
    best_candidates: list[dict[str, Any]] = []
    for lookback in (3, 24):
        rows = load_rows(args.temporal_dataset, lookback)
        arrays, preprocessing = prepare_arrays(rows, telemetry, machines, seed=42)
        (args.output_root / f"lookback_{lookback}h").mkdir()
        (args.output_root / f"lookback_{lookback}h" / "temporal_preprocessing.json").write_text(json.dumps(preprocessing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logistic = fit_logistic_validation(arrays)
        (args.output_root / f"lookback_{lookback}h" / "logistic.json").write_text(json.dumps(logistic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for architecture in ARCHITECTURES:
            for weight_mode in ("weighted", "capped20"):
                name = f"lookback_{lookback}h/{architecture}_{weight_mode}"
                result = train_candidate(arrays, architecture=architecture, weight_mode=weight_mode, seed=42, epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_root / name, device=device)
                result["lookback_hours"] = lookback; sweep.append(result)
        best = max([r for r in sweep if r["lookback_hours"] == lookback], key=lambda r: r["validation"]["macro_f1"])
        best_candidates.append(best)

    selected = max(best_candidates, key=lambda r: r["validation"]["macro_f1"])
    selected_dir = args.output_root / "selected"
    selected_dir.mkdir()
    selected_checkpoint = Path(selected["checkpoint_path"])
    shutil.copy2(selected_checkpoint, selected_dir / "selected_model.pt")
    selected_lookback = selected["lookback_hours"]
    selected_rows = load_rows(args.temporal_dataset, selected_lookback)
    selected_arrays, selected_preprocessing = prepare_arrays(selected_rows, telemetry, machines, seed=42)
    selected_prob = np.load(selected_checkpoint.parent / "probabilities.npz")
    val_labels = selected_prob["validation_labels"]; val_probs = selected_prob["validation"]
    test_labels = selected_prob["test_labels"]; test_probs = selected_prob["test"]
    thresholds, tuned_val_f1 = tune_thresholds(val_labels, val_probs)
    tuned_test_predictions = threshold_predictions(test_probs, np.asarray(thresholds))
    tuned_test_prob = np.zeros_like(test_probs); tuned_test_prob[np.arange(len(tuned_test_predictions)), tuned_test_predictions] = 1.0
    threshold_result = {"thresholds": thresholds, "validation_macro_f1": tuned_val_f1, "test_standard": metrics(test_labels, test_probs), "test_tuned": metrics(test_labels, tuned_test_prob), "test_ece_standard": expected_calibration_error(test_labels, test_probs)}
    (selected_dir / "thresholds_and_calibration.json").write_text(json.dumps(threshold_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (selected_dir / "model_card.md").write_text(f"# Azure PdM selected model\n\nLookback: {selected_lookback} hours\n\nArchitecture: {selected['architecture']}\n\nWeight mode: {selected['weight_mode']}\n\nSelection criterion: validation macro F1\n\nDataset: simulated benchmark\n\nEvent features: excluded because availability is unknown\n", encoding="utf-8")

    # Three-seed stability for the selected architecture/lookback.
    stability = []
    for seed in (7, 42, 99):
        rows = load_rows(args.temporal_dataset, selected_lookback); arrays, _ = prepare_arrays(rows, telemetry, machines, seed=seed)
        result = train_candidate(arrays, architecture=selected["architecture"], weight_mode=selected["weight_mode"], seed=seed, epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_root / f"stability_seed_{seed}", device=device)
        stability.append({"seed": seed, "validation_macro_f1": result["validation"]["macro_f1"], "test_macro_f1": result["test_standard"]["macro_f1"], "test_balanced_accuracy": result["test_standard"]["balanced_accuracy"], "checkpoint_sha256": result["checkpoint_sha256"]})
    # Machine-held-out evaluation, separate from the temporal benchmark.
    machine_rows = machine_split_rows(load_rows(args.temporal_dataset, selected_lookback))
    machine_arrays, machine_preprocessing = prepare_arrays(machine_rows, telemetry, machines, seed=42)
    machine_result = train_candidate(machine_arrays, architecture=selected["architecture"], weight_mode=selected["weight_mode"], seed=42, epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_root / "machine_held_out", device=device)
    machine_result["machine_split"] = "machines 1-60 train, 61-80 validation, 81-100 test; all timestamps retained within each machine partition"

    manifest = {
        "phase": "5.1-6.0", "contract_version": "azure_pdm_experiment_contract_v1.1", "dataset_classification": "simulated benchmark", "device": str(device), "epochs": args.epochs, "batch_size": args.batch_size,
        "derived_dataset_sha256": sha256_file(args.temporal_dataset), "source_hashes": {name: sha256_file(args.dataset_root / name) for name in ("PdM_telemetry.csv", "PdM_errors.csv", "PdM_maint.csv", "PdM_failures.csv", "PdM_machines.csv")},
        "architecture_sweep": sweep, "selected": selected, "thresholds_and_calibration": threshold_result, "seed_stability": stability, "machine_held_out": machine_result,
        "training_performed": True, "inference_performed": False, "predictive_performance_claim": False,
    }
    (args.output_root / "phase6_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "selected_architecture": selected["architecture"], "selected_lookback": selected_lookback, "selected_validation_macro_f1": selected["validation"]["macro_f1"]}, sort_keys=True))


if __name__ == "__main__":
    main()
