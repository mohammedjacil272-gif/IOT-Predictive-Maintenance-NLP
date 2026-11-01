"""Train the multimodal fusion model using aligned sensor/text records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.multimodal_fusion import MultimodalFusionModel
from training.fusion_metrics import regression_metrics
from training.fusion_provenance import (
    TRAINING_AUTHORIZATION,
    validate_genuine_approval,
    validate_named_approval,
    validate_operational_horizon,
    validate_reproducibility_contract,
    validate_verification_report,
)
from training.fusion_splits import canonical_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_integer(value, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid integer {field}: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid integer {field}: {value!r}") from exc
    if not isinstance(value, str) and value != parsed:
        raise ValueError(f"Non-integral {field}: {value!r}")
    return parsed


def validate_embedding_index_mapping(indices, embedding_count: int) -> list[int]:
    """Validate the artifact-local lookup from metadata rows to sensor vectors."""
    if isinstance(embedding_count, bool) or not isinstance(embedding_count, int) or embedding_count < 0:
        raise ValueError("embedding_count must be a non-negative integer.")
    parsed = [_metadata_integer(value, "embedding_index") for value in indices]
    if len(parsed) < embedding_count:
        present = set(parsed)
        missing = sorted(set(range(embedding_count)) - present)
        raise ValueError(
            "Missing embedding_index values (metadata/embedding count mismatch): "
            f"{missing[:5]}."
        )
    if len(parsed) > embedding_count:
        raise ValueError("Sensor embedding and metadata row counts differ.")
    if len(parsed) != len(set(parsed)):
        raise ValueError("Duplicate embedding_index values in sensor metadata.")
    out_of_range = [index for index in parsed if index < 0 or index >= embedding_count]
    if out_of_range:
        raise ValueError(f"Out-of-range embedding_index values: {out_of_range[:5]}.")
    missing = sorted(set(range(embedding_count)) - set(parsed))
    if missing:
        raise ValueError(f"Missing embedding_index values: {missing[:5]}.")
    return parsed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def load_records(
    sensor_dir: Path,
    text_file: Path,
    *,
    approval_config: dict,
    split_config: dict,
):
    """Load sensor embeddings and metadata plus a JSON list of text records."""
    if not isinstance(approval_config.get("approval_register_id"), str) or not approval_config["approval_register_id"].strip():
        raise ValueError("Genuine-data use is blocked without an approval_register_id.")
    validate_genuine_approval(approval_config)
    validate_named_approval(approval_config, TRAINING_AUTHORIZATION)
    for field in ("data_schema", "embedding_model", "embedding_revision", "prediction_cutoff", "point_in_time_policy"):
        if not approval_config.get(field):
            raise ValueError(f"Approved run metadata is missing {field}.")
    validate_operational_horizon(approval_config.get("operational_horizon"))
    validate_reproducibility_contract(approval_config)
    policies = approval_config.get("fusion_policies", {})
    if policies.get("missing_log") not in {"zero_vector", "error"}:
        raise ValueError("Approved fusion_policies.missing_log is required.")
    if policies.get("multiple_logs") not in {"mean_sorted_log_id", "error"}:
        raise ValueError("Approved fusion_policies.multiple_logs is required.")
    if approval_config.get("has_log_feature") not in {"include", "exclude"}:
        raise ValueError("Approved has_log_feature selection is required.")
    metadata_path = sensor_dir / "sensor_embedding_metadata.csv"

    import pandas as pd

    # Validate each engine against the frozen manifest before loading target
    # labels. Source-split strings alone are not sufficient provenance.
    train_ids = set(map(int, split_config.get("train_engine_ids", [])))
    val_ids = set(map(int, split_config.get("validation_engine_ids", [])))
    if (
        not train_ids
        or not val_ids
        or train_ids & val_ids
        or len(train_ids) != split_config.get("train_engine_count")
        or len(val_ids) != split_config.get("validation_engine_count")
    ):
        raise ValueError("Frozen split manifest is missing, inconsistent, or has overlapping engine IDs.")

    metadata_sha_before = sha256_file(metadata_path)
    split_metadata = pd.read_csv(
        metadata_path,
        usecols=["source_split", "unit_id"],
    )

    if not split_metadata["source_split"].isin(
        ["train", "validation"]
    ).all():
        raise ValueError(
            "Unexpected split found in sensor metadata. "
            "Only train and validation embeddings are permitted."
        )
    guarded_unit_ids = []
    for raw_unit_id, source_split in zip(
        split_metadata["unit_id"], split_metadata["source_split"]
    ):
        unit_id = _metadata_integer(raw_unit_id, "unit_id")
        if unit_id <= 0:
            raise ValueError(f"Invalid unit_id in sensor metadata: {raw_unit_id!r}")
        guarded_unit_ids.append(unit_id)
        expected_split = (
            "train" if unit_id in train_ids
            else "validation" if unit_id in val_ids
            else None
        )
        if expected_split is None or source_split != expected_split:
            raise ValueError(
                f"Sensor metadata split for unit {unit_id} is not authorized by "
                "the frozen train/validation equipment manifest."
            )

    if not isinstance(approval_config["prediction_cutoff"], dict):
        raise ValueError("Approved prediction_cutoff must map every window key to its cutoff.")

    metadata = pd.read_csv(metadata_path)
    if sha256_file(metadata_path) != metadata_sha_before:
        raise ValueError("Sensor metadata changed while split and target data were read.")
    required = {
        "unit_id", "window_start_cycle", "window_end_cycle", "label_cycle",
        "label_rul", "source_split", "namespace", "embedding_index",
        "source_dataset", "source_dataset_version", "source_record_id",
        "equipment_mapping_id", "preprocessing_version", "artifact_id",
        "embedding_model", "embedding_revision",
    }
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            "Sensor artifact lacks authoritative provenance columns: "
            f"{sorted(missing_columns)}. Genuine loading is blocked."
        )

    metadata_unit_ids = [
        _metadata_integer(value, "unit_id") for value in metadata["unit_id"]
    ]
    if metadata["source_split"].tolist() != split_metadata["source_split"].tolist() or (
        metadata_unit_ids != guarded_unit_ids
    ):
        raise ValueError("Sensor metadata changed between split and target reads.")
    if not metadata["namespace"].eq("genuine").all():
        raise ValueError("Genuine training requires explicit genuine sensor namespaces.")
    metadata["unit_id"] = metadata_unit_ids
    metadata["label_cycle"] = [
        _metadata_integer(value, "label_cycle") for value in metadata["label_cycle"]
    ]
    metadata["window_start_cycle"] = [
        _metadata_integer(value, "window_start_cycle")
        for value in metadata["window_start_cycle"]
    ]
    metadata["window_end_cycle"] = [
        _metadata_integer(value, "window_end_cycle")
        for value in metadata["window_end_cycle"]
    ]
    if (
        (metadata["window_start_cycle"] <= 0).any()
        or (metadata["window_end_cycle"] < metadata["window_start_cycle"]).any()
        or (metadata["label_cycle"] != metadata["window_end_cycle"]).any()
    ):
        raise ValueError("Sensor window bounds must be valid and label_cycle must equal window_end_cycle.")
    metadata["embedding_index"] = [
        _metadata_integer(value, "embedding_index")
        for value in metadata["embedding_index"]
    ]
    metadata["label_rul"] = pd.to_numeric(metadata["label_rul"], errors="raise")
    if metadata["label_rul"].isna().any() or not np.isfinite(
        metadata["label_rul"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing or non-finite RUL labels in sensor metadata.")
    # The frozen baseline labels every observed cycle as max_cycle(unit) - cycle.
    # Validate that contract against the sensor windows in this artifact without
    # deriving or rewriting labels in baseline preprocessing.
    max_cycle_by_unit = metadata.groupby("unit_id")["label_cycle"].transform("max")
    expected_rul = max_cycle_by_unit - metadata["label_cycle"]
    if not np.array_equal(
        metadata["label_rul"].to_numpy(dtype=float), expected_rul.to_numpy(dtype=float)
    ):
        raise ValueError(
            "Sensor artifact RUL labels do not match the frozen max_cycle(unit) - cycle target."
        )

    embeddings_path = sensor_dir / "sensor_embeddings.npy"
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(f"Unexpected sensor embedding shape: {embeddings.shape}")
    metadata["embedding_index"] = validate_embedding_index_mapping(
        metadata["embedding_index"].tolist(), int(embeddings.shape[0])
    )
    if metadata.duplicated(["unit_id", "label_cycle"]).any():
        raise ValueError("Duplicate sensor window metadata detected.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Sensor embeddings contain NaN or Inf.")
    embeddings_sha256 = sha256_file(embeddings_path)
    sensor_records = []
    for row in metadata.to_dict("records"):
        vector = embeddings[int(row["embedding_index"])]
        sensor_records.append({
            "unit_id": int(row["unit_id"]),
            "cycle": int(row["label_cycle"]),
            "window_start_cycle": int(row["window_start_cycle"]),
            "window_end_cycle": int(row["window_end_cycle"]),
            "split": str(row["source_split"]),
            "sensor_embedding": vector,
            "rul": float(row["label_rul"]),
            "namespace": row["namespace"],
            "source_dataset": row["source_dataset"],
            "source_dataset_version": row["source_dataset_version"],
            "source_record_id": row["source_record_id"],
            "equipment_mapping_id": row["equipment_mapping_id"],
            "preprocessing_version": row["preprocessing_version"],
            "artifact_id": row["artifact_id"],
            "artifact_sha256": embeddings_sha256,
            "embedding_model": row["embedding_model"],
            "embedding_revision": row["embedding_revision"],
        })

    with text_file.open("r", encoding="utf-8") as handle:
        text_records = json.load(handle)

    if not isinstance(text_records, list):
        raise ValueError("Text JSON must contain a list of text records.")

    for record in text_records:
        if record.get("namespace") != "genuine":
            raise ValueError("Training accepts only explicitly namespaced genuine text records.")
        for field in ("data_schema", "embedding_model", "embedding_revision"):
            if record.get(field) != approval_config[field]:
                raise ValueError(
                    f"Text record {record.get('log_id')!r} has missing or mismatched {field}."
                )

    return sensor_records, text_records



def make_dataset(samples, split_name: str, *, use_has_log: bool) -> TensorDataset:
    indices = [
        i for i, split in enumerate(samples["split"])
        if split == split_name
    ]
    if not indices:
        raise ValueError(f"No samples found for split {split_name!r}.")

    index = torch.tensor(indices, dtype=torch.long)
    tensors = [
        samples["sensor_embeddings"][index],
        samples["text_embeddings"][index],
        samples["rul"][index],
    ]
    if use_has_log:
        tensors.append(samples["has_log"][index])
    return TensorDataset(*tensors)


def evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []

    with torch.no_grad():
        for batch in loader:
            sensor, text, rul = batch[:3]
            sensor, text = sensor.to(device), text.to(device)
            has_log = batch[3].to(device) if len(batch) == 4 else None
            pred = model(sensor, text, has_log=has_log)
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
        "--approval-json", type=Path, required=True,
        help="Team-approved genuine-data contract, timestamp and policy register.",
    )
    parser.add_argument(
        "--verification-report-json", type=Path, required=True,
        help="Passing test and verification report for this exact commit/environment.",
    )
    parser.add_argument(
        "--sensor-checkpoint", type=Path, required=True,
        help="Owner-supplied checkpoint that produced the sensor embedding artifact.",
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--has-log-feature", choices=("include", "exclude"), required=True,
        help="Explicit architecture choice; production approval remains separate.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive.")
    if args.hidden_dim < 1 or args.head_dim < 1:
        raise ValueError("hidden-dim and head-dim must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")

    with args.approval_json.open("r", encoding="utf-8") as handle:
        approval_config = json.load(handle)
    if not isinstance(approval_config.get("approval_register_id"), str) or not approval_config["approval_register_id"].strip():
        raise ValueError("Genuine-data use is blocked without an approval_register_id.")
    validate_genuine_approval(approval_config)
    validate_named_approval(approval_config, TRAINING_AUTHORIZATION)
    for field in ("data_schema", "embedding_model", "embedding_revision"):
        if not approval_config.get(field):
            raise ValueError(f"Approved run metadata is missing {field}.")
    validate_operational_horizon(approval_config.get("operational_horizon"))
    validate_reproducibility_contract(approval_config)
    if not isinstance(approval_config.get("prediction_cutoff"), dict):
        raise ValueError("Approved per-window prediction_cutoff mapping is required.")
    if not approval_config.get("point_in_time_policy"):
        raise ValueError("Approved point_in_time_policy is required.")
    policies = approval_config.get("fusion_policies", {})
    if policies.get("missing_log") not in {"zero_vector", "error"}:
        raise ValueError("Approved fusion_policies.missing_log is required.")
    if policies.get("multiple_logs") not in {"mean_sorted_log_id", "error"}:
        raise ValueError("Approved fusion_policies.multiple_logs is required.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot identify the source Git commit for this run.") from exc
    if not git_commit:
        raise RuntimeError("Cannot identify the source Git commit for this run.")
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if git_status.strip():
        raise RuntimeError(
            "Genuine fusion training requires a clean, committed source tree so "
            "the recorded Git SHA identifies the executed code."
        )
    with args.verification_report_json.open("r", encoding="utf-8") as handle:
        verification_report = json.load(handle)
    if not isinstance(verification_report, dict):
        raise ValueError("Verification report JSON must be an object.")
    validate_verification_report(
        verification_report,
        git_commit=git_commit,
        python_version=platform.python_version(),
        python_executable=sys.executable,
    )
    if not args.sensor_checkpoint.is_file():
        raise FileNotFoundError(
            "The owner-supplied sensor checkpoint is unavailable; do not regenerate "
            "the frozen baseline checkpoint as a workaround."
        )
    sensor_checkpoint_sha256 = sha256_file(args.sensor_checkpoint)
    declared_checkpoint_sha256 = approval_config["checkpoint_lineage"].get(
        "source_checkpoint_sha256"
    )
    if declared_checkpoint_sha256 and declared_checkpoint_sha256.lower() != sensor_checkpoint_sha256:
        raise ValueError("Sensor checkpoint hash does not match the approved lineage record.")

    if approval_config.get("has_log_feature") != args.has_log_feature:
        raise ValueError("CLI has_log_feature differs from the approved manifest.")
    split_config_path = Path(
        "artifacts/data/fd001_lstm_baseline/split_config.json"
    )
    with split_config_path.open("r", encoding="utf-8") as handle:
        split_config = json.load(handle)
    sensor_records, text_records = load_records(
        args.sensor_dir,
        args.text_json,
        approval_config=approval_config,
        split_config=split_config,
    )
    # Verify sensor records against the frozen engine split.
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

    samples = build_fusion_samples(
        sensor_records,
        text_records,
        missing_log_policy=policies["missing_log"],
        multiple_log_policy=policies["multiple_logs"],
        genuine_approval=approval_config,
        point_in_time_policy=approval_config["point_in_time_policy"],
        prediction_cutoff=approval_config["prediction_cutoff"],
    )

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
        make_dataset(samples, "train", use_has_log=args.has_log_feature == "include"),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        make_dataset(samples, "validation", use_has_log=args.has_log_feature == "include"),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_config = {
        "sensor_dim": 64,
        "text_dim": 768,
        "hidden_dim": args.hidden_dim,
        "head_dim": args.head_dim,
        "dropout": args.dropout,
        "use_has_log": args.has_log_feature == "include",
    }
    model = MultimodalFusionModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    optimizer_config = {
        "name": "Adam",
        "learning_rate": args.learning_rate,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    dependency_versions = {}
    for package in (
        "torch", "numpy", "pandas", "scikit-learn", "pytest", "joblib",
        "transformers", "tokenizers",
    ):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None
    fusion_split_manifest = canonical_manifest({
        "train": list(map(int, split_config["train_engine_ids"])),
        "validation": list(map(int, split_config["validation_engine_ids"])),
    }, namespace="genuine", expected_equipment_ids=list(
        map(int, split_config["train_engine_ids"])
    ) + list(map(int, split_config["validation_engine_ids"])))
    input_paths = {
        "sensor_embeddings": args.sensor_dir / "sensor_embeddings.npy",
        "sensor_metadata": args.sensor_dir / "sensor_embedding_metadata.csv",
        "technician_records": args.text_json,
        "approval_register": args.approval_json,
        "baseline_split_manifest": split_config_path,
        "baseline_preprocessing_config": Path(
            "artifacts/data/fd001_lstm_baseline/baseline_preprocessing_config.json"
        ),
        "baseline_training_config": Path(
            "artifacts/models/fd001_lstm_baseline/training_config.json"
        ),
        "baseline_model_config": Path(
            "artifacts/models/fd001_lstm_baseline/model_config.json"
        ),
        "verification_report": args.verification_report_json,
        "sensor_checkpoint": args.sensor_checkpoint,
    }
    run_metadata = {
        "git_commit": git_commit,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "fusion_split_manifest": fusion_split_manifest,
        "fusion_split_manifest_sha256": fusion_split_manifest["sha256"],
        "python_executable": sys.executable,
        "command": list(sys.argv),
        "test_results": verification_report,
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "architecture": "concatenate_sensor_text_mlp_regression",
        "model_config": model_config,
        "training_configuration": {
            "optimizer": optimizer_config,
            "loss": "MSELoss",
            "batch_size": args.batch_size,
            "train_shuffle": True,
            "validation_shuffle": False,
            "epochs_requested": args.epochs,
            "early_stopping_patience": args.patience,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "seed": args.seed,
        },
        "optimizer": optimizer_config,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "early_stopping_patience": args.patience,
        "loss": "MSELoss",
        "seed": args.seed,
        "random_seeds": {
            "python": args.seed,
            "numpy": args.seed,
            "torch": args.seed,
            "cuda_all_devices": args.seed if torch.cuda.is_available() else None,
        },
        "deterministic_algorithms": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "device": str(device),
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions,
        "preprocessing_version": approval_config["preprocessing_version"],
        "nlp_model": {
            "name": approval_config["nlp_model_name"],
            "revision": approval_config["nlp_model_revision"],
            "sha256": approval_config.get("nlp_model_sha256"),
        },
        "tokenizer": {
            "name": approval_config["tokenizer_name"],
            "revision": approval_config["tokenizer_revision"],
            "sha256": approval_config.get("tokenizer_sha256"),
        },
        "checkpoint_lineage": approval_config["checkpoint_lineage"],
        "sensor_checkpoint_sha256": sensor_checkpoint_sha256,
        "operational_horizon": validate_operational_horizon(
            approval_config["operational_horizon"]
        ),
        "data_schema": approval_config["data_schema"],
        "embedding_model": approval_config["embedding_model"],
        "embedding_revision": approval_config["embedding_revision"],
        "prediction_cutoff": approval_config["prediction_cutoff"],
        "point_in_time_policy": approval_config["point_in_time_policy"],
        "fusion_policies": policies,
        "approval_register_id": approval_config["approval_register_id"],
        "approvals": approval_config["approvals"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            sensor, text, rul = batch[:3]
            sensor, text, rul = (
                sensor.to(device), text.to(device), rul.to(device)
            )
            has_log = batch[3].to(device) if len(batch) == 4 else None
            optimizer.zero_grad(set_to_none=True)
            predictions = model(sensor, text, has_log=has_log)
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
                "model_config": model_config,
                "run_metadata": run_metadata,
                "epochs_completed": epoch,
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
            "model_config": model_config,
            "official_test_evaluated": False,
            "text_input": str(args.text_json),
            "epochs_completed": len(history),
            "run_metadata": run_metadata,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation RMSE: {best_rmse:.4f}")
    print("Official test set was not evaluated.")


if __name__ == "__main__":
    main()
