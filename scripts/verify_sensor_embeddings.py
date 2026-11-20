"""Read-only validation of the checked-in sensor embedding artifact."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from training.train_multimodal import validate_embedding_index_mapping


def verify_sensor_artifact(artifact_dir: Path) -> dict[str, object]:
    array_path = artifact_dir / "sensor_embeddings.npy"
    metadata_path = artifact_dir / "sensor_embedding_metadata.csv"
    embeddings = np.load(array_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path)

    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(f"Expected a [rows, 64] sensor array, got {embeddings.shape}.")
    if not np.issubdtype(embeddings.dtype, np.floating) or not np.isfinite(embeddings).all():
        raise ValueError("Sensor embeddings must be finite floating-point values.")
    required = {
        "unit_id", "window_start_cycle", "window_end_cycle", "label_cycle",
        "label_rul", "source_split", "embedding_index", "embedding_source",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Sensor metadata is missing columns: {sorted(missing)}")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError("Sensor metadata and array row counts differ.")
    validate_embedding_index_mapping(metadata["embedding_index"].tolist(), int(embeddings.shape[0]))
    if metadata.duplicated(["unit_id", "label_cycle"]).any():
        raise ValueError("Duplicate (unit_id, label_cycle) sensor mappings detected.")
    if (metadata["window_start_cycle"] <= 0).any() or (
        metadata["window_end_cycle"] < metadata["window_start_cycle"]
    ).any() or (metadata["window_end_cycle"] != metadata["label_cycle"]).any():
        raise ValueError("Sensor window/cycle alignment is invalid.")
    if not metadata["source_split"].isin(["train", "validation"]).all():
        raise ValueError("Unexpected sensor source split.")
    if not metadata["embedding_source"].isin(["train", "validation"]).all():
        raise ValueError("Unexpected sensor embedding source.")
    if not (metadata["source_split"] == metadata["embedding_source"]).all():
        raise ValueError("Sensor source split and embedding source disagree.")

    return {
        "rows": int(len(metadata)),
        "dimension": int(embeddings.shape[1]),
        "index_min": int(metadata["embedding_index"].min()),
        "index_max": int(metadata["embedding_index"].max()),
        "duplicate_keys": int(metadata.duplicated(["unit_id", "label_cycle"]).sum()),
        "split_counts": {str(k): int(v) for k, v in metadata["source_split"].value_counts().sort_index().items()},
    }


def main() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts" / "embeddings" / "fd001_lstm_baseline"
    result = verify_sensor_artifact(artifact_dir)
    print("Sensor embedding artifact verification passed.")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("No files were modified.")


if __name__ == "__main__":
    main()
