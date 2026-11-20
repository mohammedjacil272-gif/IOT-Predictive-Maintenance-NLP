from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed"
OUTPUT_DIR = ROOT / "models" / "lstm_run_001"

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SensorLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1, :]).squeeze(1)


def load_split(name):
    path = DATA_DIR / (name + ".npz")
    with np.load(path) as data:
        X = data["X"].copy()
        y = data["y"].copy()
        times = data["endpoint_times"].copy()
        assets = data["endpoint_assets"].copy()

    if X.ndim != 3 or X.shape[1:] != (30, 5):
        raise ValueError(f"Unexpected X shape in {name}: {X.shape}")
    if y.shape != (len(X),):
        raise ValueError(f"Unexpected y shape in {name}: {y.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features in {name}")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Invalid labels in {name}")

    return X, y, times, assets


def predict_probabilities(model, X, device):
    model.eval()
    results = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            logits = model(batch)
            results.extend(torch.sigmoid(logits).cpu().numpy())
    return np.asarray(results)


def choose_validation_threshold(y, probabilities):
    candidates = np.unique(
        np.concatenate(([0.0], probabilities, [1.0]))
    )
    best = None

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(y, predictions, zero_division=0)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)

        candidate = {
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

        if best is None or candidate["f1"] > best["f1"]:
            best = candidate

    return best


def evaluate(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel().tolist(),
        "positive_count": int(y.sum()),
        "negative_count": int((y == 0).sum()),
        "sample_count": int(len(y)),
    }


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_DIR}"
        )

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training stopped.")

    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(0))

    X_train, y_train, _, _ = load_split("train")
    X_val, y_val, _, _ = load_split("validation")
    X_test, y_test, _, _ = load_split("test")

    if len(np.unique(y_val)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("Validation and test must each contain both classes.")

    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        raise ValueError("Training split contains no positive labels.")

    pos_weight_value = negatives / positives

    print("Training samples:", len(y_train))
    print("Training positives:", positives)
    print("Training negatives:", negatives)
    print("Positive class weight:", round(pos_weight_value, 4))
    print("Validation samples:", len(y_val))
    print("Test samples:", len(y_test))

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )

    model = SensorLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight_value], dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    config = {
        "model": "two-layer LSTM binary classifier",
        "input_features": 5,
        "window_hours": 30,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 1e-5,
        "seed": SEED,
        "positive_class_weight": pos_weight_value,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "target": "failure_next_24h",
        "threshold_selection": "maximum validation F1",
        "test_used_for_training_or_selection": False,
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    best_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)

        train_loss = total_loss / total_samples
        val_probabilities = predict_probabilities(model, X_val, device)
        val_ap = average_precision_score(y_val, val_probabilities)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_average_precision": float(val_ap),
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"validation_AP={val_ap:.5f}"
        )

        if val_ap > best_ap:
            best_ap = float(val_ap)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_average_precision": best_ap,
                    "config": config,
                },
                OUTPUT_DIR / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv", index=False
    )

    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    val_probabilities = predict_probabilities(model, X_val, device)
    selected_threshold = choose_validation_threshold(
        y_val, val_probabilities
    )

    print("\nSelected validation threshold:", selected_threshold["threshold"])
    print("Validation F1:", selected_threshold["f1"])

    # Test is evaluated only once, after model and threshold selection.
    test_probabilities = predict_probabilities(model, X_test, device)
    test_metrics = evaluate(
        y_test, test_probabilities, selected_threshold["threshold"]
    )
    validation_metrics = evaluate(
        y_val, val_probabilities, selected_threshold["threshold"]
    )

    results = {
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_ap,
        "validation_threshold_selection": selected_threshold,
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_metrics": test_metrics,
        "test_evaluated_once_after_selection": True,
        "dataset_type": "synthetic",
        "industrial_deployment_validated": False,
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\n========== FINAL TEST METRICS ==========")
    print(json.dumps(test_metrics, indent=2))
    print("\nArtifacts saved to:", OUTPUT_DIR)
    print("Training complete.")


if __name__ == "__main__":
    main()
