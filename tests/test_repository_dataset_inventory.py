
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_repository_datasets import (
    DatasetValidationError,
    _check_array,
    _read_json,
    build_inventory,
)


def test_inventory_keeps_fd001_and_synthetic_fixture_separate():
    report = build_inventory(Path(__file__).parents[1])
    datasets = {
        item["descriptor"]["namespace"]: item
        for item in report["datasets"]
    }

    assert report["validation_status"] == "PASS"
    assert datasets["fd001_benchmark"]["descriptor"]["classification"] == "benchmark"
    assert datasets["fd001_benchmark"]["suitability"]["nlp"] is False
    assert datasets["fd001_nlp_fixture"]["descriptor"]["classification"] == "synthetic_fixture"
    assert datasets["fd001_nlp_fixture"]["suitability"]["genuine_multimodal"] is False
    assert report["policy"]["cross_dataset_pairing_inferred"] is False


def test_missing_array_fails_validation(tmp_path: Path):
    with pytest.raises(DatasetValidationError, match="missing"):
        _check_array(tmp_path, "missing.npy", (2, 3))


def test_incorrect_array_shape_fails_validation(tmp_path: Path):
    np.save(tmp_path / "sample.npy", np.zeros((2, 2)))

    with pytest.raises(DatasetValidationError, match="expected shape"):
        _check_array(tmp_path, "sample.npy", (2, 3))


def test_non_finite_array_fails_validation(tmp_path: Path):
    np.save(tmp_path / "sample.npy", np.array([[1.0, np.nan]]))

    with pytest.raises(DatasetValidationError, match="non-finite"):
        _check_array(tmp_path, "sample.npy", (1, 2))


def test_malformed_json_report_fails_validation(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Cannot read JSON"):
        _read_json(tmp_path, "report.json")


def test_json_report_must_be_an_object(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Expected JSON object"):
        _read_json(tmp_path, "report.json")
