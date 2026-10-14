"""Train the multimodal fusion model using aligned sensor/text records."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.multimodal_fusion import MultimodalFusionModel
from training.fusion_metrics import regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def load_records(sensor_dir: Path, text_file: Path):
    """Load sensor embeddings and metadata plus a JSON list of text records."""
    embeddings = np.load(sensor_dir / "sensor_embeddings.npy")
    metadata_path = sensor_dir / "sensor_embedding_metadata.csv"

    import pandas as pd

    metadata = pd.read_csv(metadata_path)

    if len(embeddings) != len(metadata):
        raise ValueError("Sensor embeddings and metadata row counts differ.")
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(f"Unexpected sensor embedding shape: {embeddings.shape}")

    # Validate inputs without silently discarding records.
    if not metadata["source_split"].isin(["train", "validation"]).all():
        raise ValueError(
            "Unexpected split found in sensor metadata. "
            "Only train and validation embeddings are permitted."
        )

    if metadata.duplicated(["unit_id", "label_cycle"]).any():
        raise ValueError("Duplicate sensor window metadata detected.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Sensor embeddings contain NaN or Inf.")
    if metadata["label_rul"].isna().any():
        raise ValueError("Missing RUL labels in sensor metadata.")

    sensor_records = []
    for vector, row in zip(embeddings, metadata.to_dict("records")):
        sensor_records.append({
            "unit_id": int(row["unit_id"]),
            "cycle": int(row["label_cycle"]),
            "split": str(row["source_split"]),
            "sensor_embedding": vector,
            "rul": float(row["label_rul"]),
        })

    with text_file.open("r", encoding="utf-8") as handle:
        text_records = json.load(handle)

    if not isinstance(text_records, list):
        raise ValueError("Text JSON must contain a list of text records.")

    return sensor_records, text_records



def make_dataset(samples, split_name: str) -> TensorDataset:
    indices = [
        i for i, split in enumerate(samples["split"])
        if split == split_name
    ]
    if not indices:
        raise ValueError(f"No samples found for split {split_name!r}.")

    index = torch.tensor(indices, dtype=torch.long)
    return TensorDataset(
        samples["sensor_embeddings"][index],
        samples["text_embeddings"][index],
        samples["rul"][index],
    )


def evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []

    with torch.no_grad():
        for sensor, text, rul in loader:
            sensor, text = sensor.to(device), text.to(device)
            pred = model(sensor, text)
            targets.extend(rul.numpy().tolist())
            predictions.extend(pred.cpu().numpy().tolist())

    return regression_metrics(targets, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor-dir",
        type=Path,
        default=Path("artifacts/embeddings/fd001_lstm_baseline"),
    )
    parser.add_argument(
        "--text-json",
        type=Path,
        required=True,
        help="JSON list of real text records; see the task interface.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/fd001_multimodal"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_records, text_records = load_records(args.sensor_dir, args.text_json)
    # Verify sensor records against the frozen engine split.
    split_config_path = Path(
        "artifacts/data/fd001_lstm_baseline/split_config.json"
    )
    with split_config_path.open("r", encoding="utf-8") as handle:
        split_config = json.load(handle)

    train_ids = set(split_config["train_engine_ids"])
    val_ids = set(split_config["validation_engine_ids"])

    if train_ids & val_ids:
        raise ValueError("Frozen train/validation engine IDs overlap.")

    if len(train_ids) != split_config["train_engine_count"]:
        raise ValueError("Frozen training engine count does not match config.")

    if len(val_ids) != split_config["validation_engine_count"]:
        raise ValueError("Frozen validation engine count does not match config.")

    for record in sensor_records:
        unit_id = record["unit_id"]
        expected_split = (
            "train" if unit_id in train_ids
            else "validation" if unit_id in val_ids
            else None
        )

        if expected_split is None:
            raise ValueError(
                f"Engine {unit_id} is absent from the frozen split."
            )

        if record["split"] != expected_split:
            raise ValueError(
                f"Engine {unit_id} has split {record['split']!r}; "
                f"expected {expected_split!r}."
            )

    print("Frozen engine split verification: PASS")


    from training.fusion_dataset import build_fusion_samples

    samples = build_fusion_samples(sensor_records, text_records)

    # Confirm that no engine appears in both splits.
    train_engines = {
        int(samples["unit_id"][i])
        for i, split in enumerate(samples["split"]) if split == "train"
    }
    val_engines = {
        int(samples["unit_id"][i])
        for i, split in enumerate(samples["split"]) if split == "validation"
    }
    if train_engines & val_engines:
        raise ValueError("Train/validation engine overlap detected.")

    train_loader = DataLoader(
        make_dataset(samples, "train"),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        make_dataset(samples, "validation"),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MultimodalFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for sensor, text, rul in train_loader:
            sensor, text, rul = (
                sensor.to(device), text.to(device), rul.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(sensor, text)
            loss = criterion(predictions, rul)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss detected.")

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "validation": metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train MSE {record['train_mse']:.4f}"
            f" | val MAE {metrics['MAE']:.4f}"
            f" | val RMSE {metrics['RMSE']:.4f}"
            f" | val MSE {metrics['MSE']:.4f}"
        )

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_metrics": metrics,
                "seed": args.seed,
            }, args.output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "best_epoch": best_epoch,
            "best_validation_rmse": best_rmse,
            "device": str(device),
            "seed": args.seed,
            "official_test_evaluated": False,
            "text_input": str(args.text_json),
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation RMSE: {best_rmse:.4f}")
    print("Official test set was not evaluated.")


if __name__ == "__main__":
    main()
