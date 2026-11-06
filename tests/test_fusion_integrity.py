
import pytest
import torch

from training.fusion_integrity import (
    RECEIPT_KEY,
    issue_integrity_receipt,
    verify_integrity_receipt,
)


@pytest.fixture
def sample_data():
    return {
        "unit_id": torch.tensor([1, 2]),
        "cycle": torch.tensor([10, 20]),
        "sensor_embeddings": torch.randn(2, 64),
        "text_embeddings": torch.randn(2, 768),
        "rul": torch.tensor([100.0, 80.0]),
        "has_log": torch.tensor([1.0, 0.0]),
        "namespace": "test_namespace",
        "split": ("train", "validation"),
        "log_ids_by_window": [["log1"], []],
        "entities_by_window": [["machine1"], ["machine2"]],
        "severity_by_log_by_window": [[{"log1": 2}], [{}]],
        "timestamps_by_log_by_window": [[{"log1": "2026-01-01"}], [{}]],
        "raw_timestamps_by_log_by_window": [[{"log1": "raw"}], [{}]],
        "source_ids_by_window": [["source1"], ["source2"]],
        "sensor_provenance_by_window": [["sensor1"], ["sensor2"]],
        "window_severity": [2, 0],
        "excluded_logs": [],
    }


def test_receipt_creation_and_verification(sample_data):
    issue_integrity_receipt(sample_data)

    assert RECEIPT_KEY in sample_data
    verify_integrity_receipt(sample_data)


def test_modified_sensor_embeddings_are_detected(sample_data):
    issue_integrity_receipt(sample_data)

    sample_data["sensor_embeddings"][0, 0] += 1

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_integrity_receipt(sample_data)


def test_modified_metadata_is_detected(sample_data):
    issue_integrity_receipt(sample_data)

    sample_data["namespace"] = "modified_namespace"

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_integrity_receipt(sample_data)


def test_missing_receipt_is_rejected(sample_data):
    with pytest.raises(ValueError, match="receipt is missing"):
        verify_integrity_receipt(sample_data)


def test_invalid_receipt_algorithm_is_rejected(sample_data):
    issue_integrity_receipt(sample_data)

    sample_data[RECEIPT_KEY]["algorithm"] = "md5"

    with pytest.raises(ValueError, match="algorithm"):
        verify_integrity_receipt(sample_data)


def test_invalid_receipt_version_is_rejected(sample_data):
    issue_integrity_receipt(sample_data)

    sample_data[RECEIPT_KEY]["version"] = 999

    with pytest.raises(ValueError, match="version"):
        verify_integrity_receipt(sample_data)


def test_receipt_cannot_be_issued_twice(sample_data):
    issue_integrity_receipt(sample_data)

    with pytest.raises(ValueError, match="already contain"):
        issue_integrity_receipt(sample_data)
