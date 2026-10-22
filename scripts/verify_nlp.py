
"""Verify generated NLP artifacts without modifying the sensor baseline."""

import csv
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data" / "fd001_nlp_fixture"
EMBED_DIR = ROOT / "artifacts" / "embeddings" / "fd001_nlp_fixture"


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def read_csv(path):
    check(path.exists(), f"File exists: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    print("Verifying synthetic NLP artifacts...\n")

    # Discover the CSV outputs created by the pipeline.
    data_csvs = list(DATA_DIR.glob("*.csv"))
    check(bool(data_csvs), "At least one data CSV exists")

    for path in data_csvs:
        rows = read_csv(path)
        print(f"  {path.name}: {len(rows)} rows")

    # Check saved embedding tensors.
    tensor_files = list(EMBED_DIR.glob("*.pt"))
    check(bool(tensor_files), "At least one embedding tensor exists")

    for path in tensor_files:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        check(
            isinstance(tensor, torch.Tensor),
            f"{path.name} contains a tensor",
        )
        check(
            tensor.ndim == 2 and tensor.shape[1] == 768,
            f"{path.name} has 768-dimensional embeddings",
        )
        check(
            bool(torch.isfinite(tensor).all()),
            f"{path.name} contains only finite values",
        )
        print(f"  {path.name}: shape={tuple(tensor.shape)}")

    # Check JSON metadata if the pipeline generated any.
    json_files = list(EMBED_DIR.glob("*.json"))
    for path in json_files:
        with path.open("r", encoding="utf-8-sig") as file:
            json.load(file)
        print(f"PASS: {path.name} contains valid JSON")

    print("\nVerification completed successfully.")


if __name__ == "__main__":
    main()
