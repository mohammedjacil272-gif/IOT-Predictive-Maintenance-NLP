from pathlib import Path

import pytest

from preprocessing.dataset_registry import (
    DatasetContractError,
    DatasetDescriptor,
    inventory_files,
    validate_record_namespace,
    validate_unique_keys,
)


def test_descriptor_and_inventory_are_deterministic(tmp_path: Path):
    descriptor = DatasetDescriptor(
        namespace="example_benchmark",
        version="v1",
        classification="benchmark",
        source="https://example.invalid/dataset",
        license="research-only",
    )
    (tmp_path / "b.csv").write_text("b", encoding="utf-8")
    (tmp_path / "a.csv").write_text("a", encoding="utf-8")
    assert descriptor.to_dict()["classification"] == "benchmark"
    assert [item["relative_path"] for item in inventory_files(tmp_path)] == ["a.csv", "b.csv"]


def test_namespace_and_duplicate_keys_are_rejected():
    descriptor = DatasetDescriptor("example_benchmark", "v1", "benchmark", "source", "license")
    with pytest.raises(DatasetContractError, match="dataset_namespace"):
        validate_record_namespace({"dataset_namespace": "other"}, descriptor)
    with pytest.raises(DatasetContractError, match="duplicate"):
        validate_unique_keys([{"id": "x"}, {"id": "x"}], ("id",))


def test_invalid_descriptor_and_symlink_are_rejected(tmp_path: Path):
    with pytest.raises(DatasetContractError):
        DatasetDescriptor("FD001", "v1", "benchmark", "source", "license")
    target = tmp_path / "raw.csv"
    target.write_text("raw", encoding="utf-8")
    link = tmp_path / "link.csv"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(DatasetContractError, match="symlinks"):
        inventory_files(tmp_path)
