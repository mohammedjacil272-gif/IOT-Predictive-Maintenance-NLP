from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sensor_readings.csv"
OUT = ROOT / "processed"
EXPECTED_HASH = "9C7B6551E797A4CAF28AAE4399F3E1930BBA7BE0E5040E8F15A19BA5EA84AFDD"

EXPECTED = {
    "train": (25410, 460),
    "validation": (12957, 230),
    "test": (12979, 207),
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("SOURCE HASH")
    actual_hash = sha256(DATA)
    print("Actual:", actual_hash)
    print("Matches baseline:", actual_hash.lower() == EXPECTED_HASH.lower())

    if actual_hash.lower() != EXPECTED_HASH.lower():
        raise RuntimeError("Source dataset hash mismatch.")

    print("\nARTIFACT CHECKS")
    for name, (expected_count, expected_positive) in EXPECTED.items():
        path = OUT / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]
            times = data["endpoint_times"]
            assets = data["endpoint_assets"]

            assert X.shape == (expected_count, 30, 5), X.shape
            assert y.shape == (expected_count,), y.shape
            assert len(times) == expected_count
            assert len(assets) == expected_count
            assert int(y.sum()) == expected_positive
            assert np.isfinite(X).all()
            assert set(np.unique(y)).issubset({0, 1})

            print(
                f"{name}: PASS | X={X.shape} | "
                f"positive={int(y.sum())} | "
                f"negative={int((y == 0).sum())} | "
                f"finite=True"
            )

    metadata_path = OUT / "preprocessing_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["source_sha256"].lower() == EXPECTED_HASH.lower()
    assert metadata["model_trained"] is False
    assert metadata["source_dataset_modified"] is False

    print("\nMetadata: PASS")
    print("All three split artifacts: PASS")
    print("Source dataset unchanged: PASS")
    print("No model trained: PASS")

if __name__ == "__main__":
    main()
