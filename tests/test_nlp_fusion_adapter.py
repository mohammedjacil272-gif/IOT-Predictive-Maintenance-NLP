"""Synthetic contract tests for the Rafaz NLP to fusion schema boundary."""

import pytest
import torch

from training.fusion_dataset import build_fusion_samples
from training.fusion_provenance import parse_aware_timestamp
from training.nlp_fusion_adapter import adapt_nlp_embedding_metadata


def metadata_row(index, log_id, unit_id, cycle):
    return {
        "row_index": index,
        "log_id": log_id,
        "unit_id": str(unit_id),
        "cycle": str(cycle),
        "source_split": "fixture",
        "model_name": "distilbert-base-uncased",
        "max_length": "128",
        "pooling": "attention-mask mean pooling",
        "embedding_dim": "768",
    }


def sensor_record(unit_id, cycle):
    return {
        "unit_id": unit_id,
        "cycle": cycle,
        "split": "fixture",
        "sensor_embedding": [0.25] * 64,
        "rul": 1.0,
        "namespace": "synthetic_fixture",
    }


def processed_from(rows):
    return [
        {
            "log_id": row["log_id"],
            "unit_id": row["unit_id"],
            "cycle": row["cycle"],
            "source_split": row["source_split"],
        }
        for row in rows
    ]


def report_for(rows, embeddings):
    first = rows[0]
    return {
        "dataset_type": "synthetic_fixture",
        "number_of_logs": len(rows),
        "embedding_shape": list(embeddings.shape),
        "embedding_finite": bool(torch.isfinite(embeddings).all().item()),
        "official_test_evaluation_performed": False,
        "model_name": first["model_name"],
        "max_length": int(first["max_length"]),
        "pooling": first["pooling"],
    }


def adapt(rows, embeddings, *, processed_logs=None, verification_report=None):
    return adapt_nlp_embedding_metadata(
        rows,
        embeddings,
        processed_logs=processed_logs if processed_logs is not None else processed_from(rows),
        verification_report=(
            verification_report
            if verification_report is not None
            else report_for(rows, embeddings)
        ),
    )


def test_rafaz_schema_maps_split_and_row_index_without_positional_join():
    rows = [
        metadata_row(0, "L-A", 12, 45),
        metadata_row(1, "L-B", 12, 45),
        metadata_row(2, "L-C", 13, 46),
    ]
    vectors = torch.stack([
        torch.full((768,), 2.0),
        torch.full((768,), 4.0),
        torch.full((768,), 8.0),
    ])

    records = adapt(rows, vectors)

    assert [record["log_id"] for record in records] == ["L-A", "L-B", "L-C"]
    assert [record["split"] for record in records] == ["fixture"] * 3
    assert [record["namespace"] for record in records] == ["synthetic_fixture"] * 3
    assert [record["text_embedding"][0].item() for record in records] == [2.0, 4.0, 8.0]

    # Explicit policy arguments are test choices for synthetic records only.
    samples = build_fusion_samples(
        [sensor_record(12, 45), sensor_record(13, 46)],
        records,
        missing_log_policy="zero_vector",
        multiple_log_policy="mean_sorted_log_id",
    )
    assert samples["log_ids_by_window"] == [["L-A", "L-B"], ["L-C"]]
    assert samples["text_embeddings"][0, 0].item() == 3.0


def test_metadata_row_index_is_bound_to_processed_log_identity():
    rows = [metadata_row(0, "L-A", 1, 10), metadata_row(1, "L-B", 1, 11)]
    processed = processed_from(rows)
    processed[0]["log_id"], processed[1]["log_id"] = "L-B", "L-A"
    vectors = torch.zeros((2, 768))

    with pytest.raises(ValueError, match="maps to log_id"):
        adapt(rows, vectors, processed_logs=processed)


@pytest.mark.parametrize(
    ("rows", "embeddings", "message"),
    [
        ([metadata_row(0, "L1", 1, 1)], torch.zeros((1, 767)), "shape"),
        ([{**metadata_row(0, "L1", 1, 1), "embedding_dim": "767"}], torch.zeros((1, 768)), "embedding_dim"),
        ([metadata_row(0, "L1", 1, 1)], torch.full((1, 768), float("nan")), "non-finite"),
        ([metadata_row(0, "L1", 1, 1)], torch.full((1, 768), float("inf")), "non-finite"),
        ([metadata_row(0, "L1", 1, 1)], torch.zeros((2, 768)), "row count"),
    ],
)
def test_embedding_schema_and_tensor_integrity_are_validated(rows, embeddings, message):
    with pytest.raises(ValueError, match=message):
        adapt(rows, embeddings)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([metadata_row(0, "", 1, 1)], "log_id"),
        ([metadata_row(0, "L1", 0, 1)], "unit_id"),
        ([metadata_row(0, "L1", 1, "not-cycle")], "cycle"),
        ([metadata_row(0, "L1", 1, 1), metadata_row(1, "L1", 2, 2)], "duplicate log_id"),
        ([metadata_row(0, "L1", 1, 1), metadata_row(0, "L2", 1, 2)], "Duplicate NLP row_index"),
        ([metadata_row(0, "L1", 1, 1), metadata_row(2, "L2", 2, 2)], "out of bounds"),
    ],
)
def test_nlp_record_ids_and_row_mappings_are_validated(rows, message):
    with pytest.raises(ValueError, match=message):
        adapt(rows, torch.zeros((len(rows), 768)))


def test_split_and_required_schema_are_explicit():
    row = metadata_row(0, "L1", 1, 1)
    with pytest.raises(ValueError, match="split and source_split disagree"):
        adapt([{**row, "split": "train"}], torch.zeros((1, 768)))
    vectors = torch.zeros((1, 768))
    with pytest.raises(ValueError, match="missing fields"):
        adapt(
            [{key: value for key, value in row.items() if key != "pooling"}],
            vectors,
            verification_report=report_for([row], vectors),
        )


def test_synthetic_adapter_accepts_only_the_fixture_source_split():
    row = metadata_row(0, "L1", 1, 1)
    record = adapt([row], torch.zeros((1, 768)))[0]
    assert record["split"] == "fixture"
    assert record["namespace"] == "synthetic_fixture"


@pytest.mark.parametrize(
    ("source_split", "message"),
    [
        ("train", "must be 'fixture'"),
        ("validation", "must be 'fixture'"),
        ("test", "must be 'fixture'"),
        ("other", "must be 'fixture'"),
        ("", "invalid source_split"),
        (None, "invalid source_split"),
        (" fixture ", "surrounding whitespace"),
    ],
)
def test_synthetic_adapter_rejects_non_fixture_source_splits(source_split, message):
    row = {**metadata_row(0, "L1", 1, 1), "source_split": source_split}
    with pytest.raises(ValueError, match=message):
        adapt([row], torch.zeros((1, 768)))


def test_unavailable_source_provenance_is_not_fabricated():
    record = adapt(
        [metadata_row(0, "fixture-log", 1, 1)],
        torch.zeros((1, 768)),
    )[0]
    assert record["embedding_model"] == "distilbert-base-uncased"
    assert "embedding_revision" not in record
    assert "source_record_id" not in record
    assert "equipment_mapping_id" not in record
    assert "availability_time" not in record


def test_genuine_classification_is_rejected_by_synthetic_adapter():
    rows = [metadata_row(0, "fixture-log", 1, 1)]
    vectors = torch.zeros((1, 768))
    report = report_for(rows, vectors)
    report["dataset_type"] = "genuine"

    with pytest.raises(ValueError, match="only artifacts classified as synthetic_fixture"):
        adapt(rows, vectors, verification_report=report)

    rows[0]["namespace"] = "genuine"
    report["dataset_type"] = "synthetic_fixture"
    with pytest.raises(ValueError, match="namespace conflicts"):
        adapt(rows, vectors, verification_report=report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "other-model"),
        ("max_length", "64"),
        ("pooling", "cls-token"),
        ("tokenizer_name", "other-tokenizer"),
        ("tokenizer_revision", "different-revision"),
    ],
)
def test_mixed_embedding_configuration_is_rejected(field, value):
    first = metadata_row(0, "L-A", 1, 1)
    second = metadata_row(1, "L-B", 1, 2)
    first["tokenizer_name"] = "tokenizer-a"
    second["tokenizer_name"] = "tokenizer-a"
    if field in {"tokenizer_revision"}:
        first[field] = "revision-a"
        second[field] = "revision-a"
    second[field] = value
    rows = [first, second]
    vectors = torch.zeros((2, 768))

    with pytest.raises(ValueError, match="configuration|disagree"):
        adapt(rows, vectors)


def test_missing_or_inconsistent_upstream_verification_evidence_fails_closed():
    rows = [metadata_row(0, "L-A", 1, 1)]
    vectors = torch.zeros((1, 768))
    with pytest.raises(ValueError, match="verification report is required"):
        adapt_nlp_embedding_metadata(rows, vectors, processed_logs=processed_from(rows), verification_report=None)

    report = report_for(rows, vectors)
    report["embedding_shape"] = [1, 767]
    with pytest.raises(ValueError, match="embedding_shape does not match"):
        adapt(rows, vectors, verification_report=report)


def test_processed_metadata_key_mismatch_is_rejected():
    rows = [metadata_row(0, "L-A", 1, 1)]
    processed = processed_from(rows)
    processed[0]["cycle"] = "2"
    with pytest.raises(ValueError, match="metadata/processed-log mismatch.*cycle"):
        adapt(rows, torch.zeros((1, 768)), processed_logs=processed)


@pytest.mark.parametrize(
    ("value", "field", "message"),
    [
        ("not-a-timestamp", "available_at", "Malformed available_at"),
        ("2025-02-01T10:00:00", "available_at", "Timezone-naive available_at"),
    ],
)
def test_invalid_fixture_timestamps_are_rejected(value, field, message):
    with pytest.raises(ValueError, match=message):
        parse_aware_timestamp(value, field=field, timezone_name="UTC")
