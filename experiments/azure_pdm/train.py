"""Bounded Azure PdM model training/evaluation pipeline.

The pipeline consumes the canonical Phase 4 temporal index and raw telemetry,
fits all preprocessing on training rows only, and writes artifacts outside Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import AzureTemporalClassifier


LABELS = ("none", "comp1", "comp2", "comp3", "comp4", "multi_failure")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
TELEMETRY_FIELDS = ("volt", "rotate", "pressure", "vibration")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_telemetry(path: Path) -> dict[int, dict[datetime, tuple[float, float, float, float]]]:
    telemetry: dict[int, dict[datetime, tuple[float, float, float, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            machine = int(raw["machineID"])
            timestamp = datetime.fromisoformat(raw["datetime"])
            values = tuple(float(raw[field]) for field in TELEMETRY_FIELDS)
            telemetry.setdefault(machine, {})[timestamp] = values
    return telemetry


def load_machines(path: Path) -> dict[int, tuple[str, float]]:
    machines: dict[int, tuple[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            machines[int(raw["machineID"])] = (raw["model"], float(raw["age"]))
    return machines


def load_rows(path: Path, lookback_hours: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if int(row["lookback_hours"]) == lookback_hours]


def prepare_arrays(
    rows: list[dict[str, str]],
    telemetry: dict[int, dict[datetime, tuple[float, float, float, float]]],
    machines: dict[int, tuple[str, float]],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_indices = np.array([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    models_train = sorted({machines[int(rows[i]["machineID"])][0] for i in train_indices})
    model_to_index = {"<UNK>": 0, **{name: index + 1 for index, name in enumerate(models_train)}}
    lookback = int(rows[0]["lookback_hours"])
    x = np.empty((len(rows), lookback, len(TELEMETRY_FIELDS)), dtype=np.float32)
    metadata_raw = np.zeros((len(rows), 1 + len(model_to_index)), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    splits = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        machine = int(row["machineID"])
        anchor = datetime.fromisoformat(row["anchor_timestamp"])
        start = anchor - timedelta(hours=lookback)
        expected = [start + timedelta(hours=offset) for offset in range(lookback)]
        values = [telemetry[machine][timestamp] for timestamp in expected]
        x[index] = np.asarray(values, dtype=np.float32)
        model, age = machines[machine]
        metadata_raw[index, 0] = age
        metadata_raw[index, model_to_index.get(model, 0)] = 1.0
        labels[index] = LABEL_TO_ID[row["target_label"]]
        splits[index] = row["split"]

    telemetry_mean = x[train_indices].reshape(-1, x.shape[-1]).mean(axis=0)
    telemetry_std = x[train_indices].reshape(-1, x.shape[-1]).std(axis=0)
    telemetry_std[telemetry_std == 0] = 1.0
    x = (x - telemetry_mean.reshape(1, 1, -1)) / telemetry_std.reshape(1, 1, -1)
    age_mean = float(metadata_raw[train_indices, 0].mean())
    age_std = float(metadata_raw[train_indices, 0].std()) or 1.0
    metadata_raw[:, 0] = (metadata_raw[:, 0] - age_mean) / age_std
    arrays = {
        "telemetry": x,
        "metadata": metadata_raw,
        "labels": labels,
        "train_indices": np.where(splits == "train")[0],
        "validation_indices": np.where(splits == "validation")[0],
        "test_indices": np.where(splits == "test")[0],
    }
    preprocessing = {
        "telemetry_fields": list(TELEMETRY_FIELDS),
        "telemetry_mean_fit_on_train": telemetry_mean.tolist(),
        "telemetry_std_fit_on_train": telemetry_std.tolist(),
        "machine_model_vocab_fit_on_train": model_to_index,
        "machine_age_mean_fit_on_train": age_mean,
        "machine_age_std_fit_on_train": age_std,
        "lookback_hours": lookback,
        "label_mapping": LABEL_TO_ID,
        "seed": seed,
    }
    return arrays, preprocessing


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=np.arange(len(LABELS)), zero_division=0
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=np.arange(len(LABELS))).tolist(),
        "per_component_recall": {label: float(recall[LABEL_TO_ID[label]]) for label in LABELS[1:5]},
        "multi_failure_precision": float(precision[LABEL_TO_ID["multi_failure"]]),
        "multi_failure_recall": float(recall[LABEL_TO_ID["multi_failure"]]),
        "class_distribution": {label: int((y_true == LABEL_TO_ID[label]).sum()) for label in LABELS},
    }
    try:
        result["ovr_average_precision_macro"] = float(
            average_precision_score(np.eye(len(LABELS))[y_true], probabilities, average="macro")
        )
    except ValueError:
        result["ovr_average_precision_macro"] = None
    return result


def majority_baseline(train_y: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    majority = int(Counter(train_y.tolist()).most_common(1)[0][0])
    probabilities = np.zeros((len(test_y), len(LABELS)), dtype=np.float32)
    probabilities[:, majority] = 1.0
    result = metrics(test_y, probabilities)
    result["baseline"] = "majority_class"
    result["majority_label"] = LABELS[majority]
    return result


def logistic_baseline(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    train = arrays["train_indices"]
    test = arrays["test_indices"]
    x = arrays["telemetry"]
    meta = arrays["metadata"]
    aggregate = np.concatenate([x.mean(axis=1), x.std(axis=1), meta], axis=1)
    classifier = LogisticRegression(max_iter=100, class_weight="balanced", solver="lbfgs", random_state=42)
    classifier.fit(aggregate[train], arrays["labels"][train])
    result = metrics(arrays["labels"][test], classifier.predict_proba(aggregate[test]))
    result["baseline"] = "balanced_logistic_regression"
    return result


def train_neural(
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train, validation, test = arrays["train_indices"], arrays["validation_indices"], arrays["test_indices"]
    x = torch.from_numpy(arrays["telemetry"])
    meta = torch.from_numpy(arrays["metadata"])
    labels = torch.from_numpy(arrays["labels"])
    train_counts = np.bincount(arrays["labels"][train], minlength=len(LABELS)).astype(np.float32)
    weights = train_counts.sum() / (len(LABELS) * np.maximum(train_counts, 1.0))
    model = AzureTemporalClassifier(telemetry_channels=x.shape[2], metadata_dim=meta.shape[1], class_count=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    loader = DataLoader(TensorDataset(x[train], meta[train], labels[train]), batch_size=batch_size, shuffle=True, num_workers=0)

    def predict(indices: np.ndarray) -> tuple[float, np.ndarray]:
        model.eval(); total_loss = 0.0; collected = []
        with torch.no_grad():
            for batch_x, batch_meta, batch_y in DataLoader(TensorDataset(x[indices], meta[indices], labels[indices]), batch_size=batch_size * 2, shuffle=False, num_workers=0):
                logits = model(batch_x.to(device), batch_meta.to(device))
                total_loss += float(criterion(logits, batch_y.to(device)).item()) * len(batch_y)
                collected.append(torch.softmax(logits, dim=1).cpu().numpy())
        return total_loss / len(indices), np.concatenate(collected, axis=0)

    history = []
    best_macro = -1.0
    best_epoch = 0
    patience_left = 1
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        for batch_x, batch_meta, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device), batch_meta.to(device))
            loss = criterion(logits, batch_y.to(device))
            loss.backward(); optimizer.step()
            train_loss += float(loss.item()) * len(batch_y)
        val_loss, val_prob = predict(validation)
        val_metrics = metrics(arrays["labels"][validation], val_prob)
        record = {"epoch": epoch, "train_loss": train_loss / len(train), "validation_loss": val_loss, "validation_macro_f1": val_metrics["macro_f1"]}
        history.append(record)
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]; best_epoch = epoch; patience_left = 1
            torch.save({"state_dict": model.state_dict(), "model_config": {"telemetry_channels": x.shape[2], "metadata_dim": meta.shape[1], "class_count": len(LABELS)}, "label_mapping": LABEL_TO_ID, "best_epoch": epoch, "validation_macro_f1": best_macro, "seed": seed}, checkpoint_path)
        else:
            patience_left -= 1
            if patience_left < 0:
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    _, test_prob = predict(test)
    evaluation = metrics(arrays["labels"][test], test_prob)
    evaluation["checkpoint_best_epoch"] = best_epoch
    evaluation["validation_best_macro_f1"] = best_macro
    return evaluation, {"history": history, "checkpoint_path": str(checkpoint_path), "class_weights": weights.tolist()}


def run_lookback(
    *,
    dataset_root: Path,
    temporal_dataset: Path,
    output_root: Path,
    lookback: int,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 2048,
) -> dict[str, Any]:
    set_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    output_dir = output_root / f"lookback_{lookback}h"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = load_rows(temporal_dataset, lookback)
    telemetry = load_telemetry(dataset_root / "PdM_telemetry.csv")
    machines = load_machines(dataset_root / "PdM_machines.csv")
    arrays, preprocessing = prepare_arrays(rows, telemetry, machines, seed=seed)
    np.savez_compressed(output_dir / "preprocessed_arrays.npz", **arrays)
    (output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_results = {
        "majority_class": majority_baseline(arrays["labels"][arrays["train_indices"]], arrays["labels"][arrays["test_indices"]]),
        "balanced_logistic_regression": logistic_baseline(arrays),
    }
    neural_results, training = train_neural(arrays, output_dir, seed=seed, epochs=epochs, batch_size=batch_size, device=device)
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(training["history"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_hash = sha256_file(output_dir / "best_model.pt")
    result = {
        "lookback_hours": lookback,
        "device": str(device),
        "seed": seed,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "row_counts": {split: int(len(arrays[f"{split}_indices"])) for split in ("train", "validation", "test")},
        "natural_class_distribution": {split: {label: int((arrays["labels"][arrays[f"{split}_indices"]] == LABEL_TO_ID[label]).sum()) for label in LABELS} for split in ("train", "validation", "test")},
        "training_class_weights": training["class_weights"],
        "baselines": baseline_results,
        "neural_model": neural_results,
        "checkpoint_path": str(output_dir / "best_model.pt"),
        "checkpoint_sha256": checkpoint_hash,
        "preprocessing_path": str(output_dir / "preprocessing.json"),
        "training_history_path": str(history_path),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--temporal-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[3, 24])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite phase5 output directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for lookback in args.lookbacks:
        results.append(run_lookback(dataset_root=args.dataset_root, temporal_dataset=args.temporal_dataset, output_root=args.output_root, lookback=lookback, epochs=args.epochs, batch_size=args.batch_size))
    manifest = {
        "phase": "4.1-5.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_classification": "simulated benchmark",
        "contract_version": "azure_pdm_experiment_contract_v1.1",
        "acquisition_manifest_path": str(args.dataset_root.parent / "acquisition_manifest_02.json"),
        "acquisition_manifest_sha256": sha256_file(args.dataset_root.parent / "acquisition_manifest_02.json"),
        "raw_source_hashes": {name: sha256_file(args.dataset_root / name) for name in ("PdM_telemetry.csv", "PdM_errors.csv", "PdM_maint.csv", "PdM_failures.csv", "PdM_machines.csv")},
        "derived_dataset_path": str(args.temporal_dataset),
        "derived_dataset_sha256": sha256_file(args.temporal_dataset),
        "derived_manifest_sha256": sha256_file(args.temporal_dataset.parent / "phase4_manifest.json"),
        "split_configuration": json.load((args.temporal_dataset.parent / "phase4_manifest.json").open(encoding="utf-8"))["configuration"]["split_policy"],
        "feature_names_and_order": {"telemetry": list(TELEMETRY_FIELDS), "metadata": ["age_standardized", "model_one_hot_train_vocab"]},
        "dependencies": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "lookback_results": results,
        "training_performed": True,
        "inference_performed": False,
        "predictive_performance_claim": False,
    }
    (args.output_root / "phase5_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "lookbacks": args.lookbacks, "checkpoints": [item["checkpoint_path"] for item in results]}, sort_keys=True))


if __name__ == "__main__":
    main()
