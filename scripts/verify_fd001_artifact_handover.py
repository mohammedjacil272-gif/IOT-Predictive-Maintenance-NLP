"""Read-only verification of the local FD001 baseline handover artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.lstm_baseline import LSTMBaseline


EXPECTED_SHA256 = {
    "models/fd001_lstm_baseline/best_model.pt": "21c059a7f87c151ef3f78db3a8af9e2fdd0441fb0214136fc1ae583ad5971d70",
    "models/fd001_lstm_baseline/last_model.pt": "76b00ce715d28aeadc8c812abc3f091a61ae0bd2506b529d0ce19d560814bc8f",
    "data/fd001_lstm_baseline/X_train.npy": "aa63e09450e2b1f2c0a008132de7b8c7166b0f07d12bab7623e08edf107acdc9",
    "data/fd001_lstm_baseline/X_val.npy": "b2cc889d6d68de4185e59c53143cdc2be81d429f6f45037439da8126df1a820b",
    "data/fd001_lstm_baseline/X_test.npy": "833fe9c8a4661f5ad17ccdf40955329168175bf5637918897a0b1d2f4a0e0dc8",
    "embeddings/fd001_lstm_baseline/sensor_embeddings.npy": "de60d4ff953a6efa3128e0b213d67d27a688513175318d9cb3ad9ee6ab7899fd",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(project_root: Path = ROOT) -> dict[str, object]:
    artifact_root = project_root / "artifacts"
    hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = artifact_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required handover artifact is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual} != {expected}")
        hashes[relative] = actual

    model_config_path = artifact_root / "models/fd001_lstm_baseline/model_config.json"
    preprocessing_config_path = artifact_root / "data/fd001_lstm_baseline/baseline_preprocessing_config.json"
    validation_metrics_path = artifact_root / "metrics/fd001_lstm_baseline/validation_metrics.json"
    for path in (model_config_path, preprocessing_config_path, validation_metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required handover metadata is missing: {path}")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        artifact_root / "models/fd001_lstm_baseline/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = LSTMBaseline(
        input_size=int(model_config["input_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        lstm_dropout=float(model_config["lstm_dropout"]),
        head_dropout=float(model_config["head_dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sample = np.load(artifact_root / "data/fd001_lstm_baseline/X_val.npy", mmap_mode="r", allow_pickle=False)
    sample_tensor = torch.from_numpy(np.array(sample[:1], dtype=np.float32, copy=True))
    with torch.inference_mode():
        prediction = model(sample_tensor)
    if prediction.shape != (1,) or not torch.isfinite(prediction).all().item():
        raise ValueError("Baseline handover sample inference is not finite [1].")
    return {
        "status": "PASS",
        "hashes": hashes,
        "model_input": [int(model_config["sequence_length"]), int(model_config["input_size"])],
        "sample_prediction_finite": True,
        "official_test_evaluation": "completed_final_evaluation_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify_artifacts(args.project_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    print("No files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
