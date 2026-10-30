
import pytest
import torch

from scripts.verify_nlp import validate_alignment


def make_sample():
    processed = [
        {
            "log_id": "L001",
            "unit_id": "12",
            "cycle": "45",
            "source_split": "fixture",
        },
        {
            "log_id": "L002",
            "unit_id": "12",
            "cycle": "46",
            "source_split": "fixture",
        },
    ]

    metadata = [
        {
            "row_index": "0",
            "log_id": "L001",
            "unit_id": "12",
            "cycle": "45",
            "source_split": "fixture",
            "model_name": "distilbert-base-uncased",
            "max_length": "128",
            "pooling": "attention-mask mean pooling",
            "embedding_dim": "768",
        },
        {
            "row_index": "1",
            "log_id": "L002",
            "unit_id": "12",
            "cycle": "46",
            "source_split": "fixture",
            "model_name": "distilbert-base-uncased",
            "max_length": "128",
            "pooling": "attention-mask mean pooling",
            "embedding_dim": "768",
        },
    ]

    tensor = torch.zeros((2, 768))

    return processed, metadata, tensor


def test_valid_alignment_passes():
    processed, metadata, tensor = make_sample()
    validate_alignment(processed, metadata, tensor)


def test_duplicate_log_id_is_rejected():
    processed, metadata, tensor = make_sample()
    processed[1]["log_id"] = "L001"

    with pytest.raises(AssertionError, match="duplicate"):
        validate_alignment(processed, metadata, tensor)


def test_missing_metadata_row_is_rejected():
    processed, metadata, tensor = make_sample()
    metadata.pop()

    with pytest.raises(AssertionError, match="row count"):
        validate_alignment(processed, metadata, tensor)


def test_incorrect_row_index_is_rejected():
    processed, metadata, tensor = make_sample()
    metadata[1]["row_index"] = "5"

    with pytest.raises(AssertionError, match="sequential"):
        validate_alignment(processed, metadata, tensor)


def test_metadata_log_order_mismatch_is_rejected():
    processed, metadata, tensor = make_sample()

    metadata.reverse()

    # Keep indexes sequential to reach the log-order validation.
    for index, row in enumerate(metadata):
        row["row_index"] = str(index)

    with pytest.raises(AssertionError, match="order"):
        validate_alignment(processed, metadata, tensor)


def test_unit_id_mismatch_is_rejected():
    processed, metadata, tensor = make_sample()
    metadata[0]["unit_id"] = "99"

    with pytest.raises(AssertionError, match="unit_id"):
        validate_alignment(processed, metadata, tensor)


def test_cycle_mismatch_is_rejected():
    processed, metadata, tensor = make_sample()
    metadata[0]["cycle"] = "999"

    with pytest.raises(AssertionError, match="cycle"):
        validate_alignment(processed, metadata, tensor)


def test_non_finite_embeddings_are_rejected():
    processed, metadata, tensor = make_sample()
    tensor[0, 0] = float("nan")

    with pytest.raises(AssertionError, match="non-finite"):
        validate_alignment(processed, metadata, tensor)


def test_wrong_embedding_dimension_is_rejected():
    processed, metadata, _ = make_sample()
    tensor = torch.zeros((2, 767))

    with pytest.raises(AssertionError, match="dimension"):
        validate_alignment(processed, metadata, tensor)


def test_invalid_embedding_configuration_is_rejected():
    processed, metadata, tensor = make_sample()
    metadata[0]["max_length"] = "invalid"

    with pytest.raises(AssertionError, match="configuration"):
        validate_alignment(processed, metadata, tensor)


def test_non_fixture_data_is_rejected():
    processed, metadata, tensor = make_sample()

    # Keep processed logs and metadata consistent so the fixture check is reached.
    processed[0]["source_split"] = "train"
    metadata[0]["source_split"] = "train"

    with pytest.raises(AssertionError, match="fixture"):
        validate_alignment(processed, metadata, tensor)
