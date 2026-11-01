import pytest
import torch

from models.multimodal_fusion import MultimodalFusionModel
from training.fusion_dataset import build_fusion_samples as _build_fusion_samples


def build_fusion_samples(sensor_records, text_records, **kwargs):
    # Test-only explicit fixture namespace and provisional policy selection.
    sensor_records = [dict(row, namespace=row.get("namespace", "synthetic_fixture")) for row in sensor_records]
    text_records = [dict(row, namespace=row.get("namespace", "synthetic_fixture")) for row in text_records]
    return _build_fusion_samples(
        sensor_records,
        text_records,
        missing_log_policy=kwargs.pop("missing_log_policy", "zero_vector"),
        multiple_log_policy=kwargs.pop("multiple_log_policy", "mean_sorted_log_id"),
        **kwargs,
    )


def sensor(unit, cycle, split="train"):
    return {
        "unit_id": unit,
        "cycle": cycle,
        "split": split,
        "sensor_embedding": [float(unit)] * 64,
        "rul": float(100 - cycle),
    }


def log(log_id, unit, cycle, value, split="train"):
    return {
        "log_id": log_id,
        "unit_id": unit,
        "cycle": cycle,
        "split": split,
        "text_embedding": [float(value)] * 768,
    }


def test_alignment_and_missing_text():
    result = build_fusion_samples(
        [sensor(2, 1), sensor(1, 1)],
        [log("b", 1, 1, 4), log("a", 1, 1, 2)],
    )

    assert result["unit_id"].tolist() == [1, 2]
    assert result["has_log"].tolist() == [True, False]
    assert torch.all(result["text_embeddings"][0] == 3)
    assert torch.all(result["text_embeddings"][1] == 0)
    assert result["sensor_embeddings"].shape == (2, 64)
    assert result["text_embeddings"].shape == (2, 768)


def test_duplicate_sensor_window_rejected():
    with pytest.raises(ValueError, match="Duplicate sensor"):
        build_fusion_samples([sensor(1, 1), sensor(1, 1)], [])


def test_invalid_cycle_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        build_fusion_samples([sensor(1, 0)], [])


def test_unmatched_text_rejected():
    with pytest.raises(ValueError, match="no matching sensor"):
        build_fusion_samples([sensor(1, 1)], [log("x", 9, 1, 2)])


def test_split_mismatch_rejected():
    with pytest.raises(ValueError, match="Split mismatch"):
        build_fusion_samples(
            [sensor(1, 1, "train")],
            [log("x", 1, 1, 2, "validation")],
        )


def test_equipment_split_leakage_rejected_without_text_records():
    with pytest.raises(ValueError, match="Equipment split leakage"):
        build_fusion_samples(
            [sensor(1, 1, "train"), sensor(1, 2, "validation")],
            [],
        )


def test_required_alignment_provenance_is_validated():
    text = log("log-1", 1, 1, 2)
    del text["split"]
    with pytest.raises(ValueError, match="Text record is missing fields.*split"):
        build_fusion_samples([sensor(1, 1)], [text])


def test_entity_provenance_mismatch_rejected():
    text = log("log-1", 1, 1, 2)
    text["entities"] = [{"log_id": "different-log", "label": "SYMPTOM"}]
    with pytest.raises(ValueError, match="Entity provenance mismatch.*log_id"):
        build_fusion_samples([sensor(1, 1)], [text])


@pytest.mark.parametrize("invalid_log_id", [None, "", "   ", 123])
def test_missing_or_invalid_entity_log_id_rejected(invalid_log_id):
    text = log("parent-log", 1, 1, 2)
    text["entities"] = [{"log_id": invalid_log_id, "label": "SYMPTOM"}]

    with pytest.raises(
        ValueError,
        match="Entity for log parent-log is missing a valid log_id",
    ):
        build_fusion_samples([sensor(1, 1)], [text])


def test_entity_without_log_id_rejected():
    text = log("parent-log", 1, 1, 2)
    text["entities"] = [{"label": "SYMPTOM"}]

    with pytest.raises(
        ValueError,
        match="Entity for log parent-log is missing a valid log_id",
    ):
        build_fusion_samples([sensor(1, 1)], [text])


def test_entity_with_unknown_log_id_rejected():
    text = log("parent-log", 1, 1, 2)
    text["entities"] = [{"log_id": "unknown-log", "label": "SYMPTOM"}]

    with pytest.raises(ValueError, match="Entity provenance mismatch.*log_id"):
        build_fusion_samples([sensor(1, 1)], [text])


def test_explicit_entity_log_ownership_is_preserved():
    text = log("parent-log", 1, 1, 2)
    text["entities"] = [{"log_id": "parent-log", "label": "SYMPTOM"}]

    result = build_fusion_samples([sensor(1, 1)], [text])

    entity = result["entities_by_window"][0][0]
    assert entity["log_id"] == "parent-log"
    assert entity["unit_id"] == 1
    assert entity["cycle"] == 1
    assert entity["split"] == "train"


def test_fusion_uses_supplied_uncapped_baseline_rul_label():
    record = sensor(1, 30)
    record["rul"] = 0.0  # Baseline label at its maximum-cycle window.
    result = build_fusion_samples([record], [])
    assert result["rul"].tolist() == [0.0]

def test_duplicate_log_id_rejected_across_different_cycles():
    with pytest.raises(ValueError, match="Duplicate log_id"):
        build_fusion_samples(
            [sensor(1, 1), sensor(1, 2)],
            [
                log("same-id", 1, 1, 2),
                log("same-id", 1, 2, 3),
            ],
        )


def test_multiple_logs_are_aggregated_deterministically():
    result = build_fusion_samples(
        [sensor(1, 1)],
        [
            log("log-b", 1, 1, 6),
            log("log-a", 1, 1, 2),
        ],
    )

    assert result["has_log"].tolist() == [True]
    assert torch.all(result["text_embeddings"][0] == 4)
    assert result["log_ids_by_window"] == [["log-a", "log-b"]]


def test_namespace_mismatch_rejected():
    sensor_row = dict(sensor(1, 1), namespace="synthetic_fixture")
    text_row = dict(log("x", 1, 1, 2), namespace="genuine")
    with pytest.raises(ValueError, match="namespace mismatch"):
        _build_fusion_samples(
            [sensor_row], [text_row],
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
        )


def test_ambiguous_namespaces_within_sensor_records_rejected():
    first = dict(sensor(1, 1), namespace="synthetic_fixture")
    second = dict(sensor(2, 1), namespace="genuine")
    with pytest.raises(ValueError, match="Ambiguous Sensor namespaces"):
        _build_fusion_samples(
            [first, second], [],
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
        )


def test_missing_namespace_rejected():
    with pytest.raises(ValueError, match="explicit namespace"):
        _build_fusion_samples(
            [sensor(1, 1)], [],
            missing_log_policy="zero_vector",
            multiple_log_policy="mean_sorted_log_id",
        )


def test_embedding_with_infinity_rejected():
    bad_sensor = sensor(1, 1)
    bad_sensor["sensor_embedding"][0] = float("inf")
    with pytest.raises(ValueError, match="Non-finite Sensor"):
        build_fusion_samples([bad_sensor], [])


def test_malformed_text_embedding_rejected():
    bad_text = log("bad", 1, 1, 1)
    bad_text["text_embedding"] = [float("nan")] * 768
    with pytest.raises(ValueError, match="Non-finite Text"):
        build_fusion_samples([sensor(1, 1)], [bad_text])


def test_unapproved_policies_have_no_implicit_defaults():
    with pytest.raises(ValueError, match="Select missing_log_policy explicitly"):
        _build_fusion_samples(
            [dict(sensor(1, 1), namespace="synthetic_fixture")], [],
        )

def test_text_cycle_aligns_to_sensor_label_cycle():
    sensor_record = {
        "unit_id": 12,
        "cycle": 45,  # sourced from sensor label_cycle
        "split": "validation",
        "sensor_embedding": [1.0] * 64,
        "rul": 100.0,
    }

    text_record = {
        "log_id": "L001",
        "unit_id": 12,
        "cycle": 45,
        "split": "validation",
        "text_embedding": [2.0] * 768,
    }

    result = build_fusion_samples(
        [sensor_record],
        [text_record],
    )

    assert result["unit_id"].tolist() == [12]
    assert result["cycle"].tolist() == [45]
    assert result["has_log"].tolist() == [True]
    assert torch.all(result["text_embeddings"][0] == 2)


def test_embedding_index_never_participates_in_cross_modal_alignment():
    sensor_record = sensor(12, 45)
    sensor_record["embedding_index"] = 900
    text_record = log("L-index-independent", 12, 45, 3.0)
    text_record["embedding_index"] = 0
    result = build_fusion_samples([sensor_record], [text_record])
    assert result["unit_id"].tolist() == [12]
    assert result["cycle"].tolist() == [45]
    assert result["log_ids_by_window"] == [["L-index-independent"]]


def test_supplied_rul_is_consumed_unchanged():
    sensor_record = sensor(12, 45)
    sensor_record["rul"] = 55.25
    result = build_fusion_samples([sensor_record], [])
    assert result["rul"].tolist() == [55.25]

def test_provenance_and_window_severity_are_preserved():
    result = build_fusion_samples(
        [sensor(1, 10), sensor(1, 20)],
        [
            {
                **log("log_b", 1, 10, 4.0),
                "entities": [{
    "log_id": "log_b",
    "entity_text": "overheating",
    "label": "SYMPTOM",
}],
                "severity_score": 4,
                "severity_label": "high",
            },
            {
                **log("log_a", 1, 10, 2.0),
                "entities": [{
    "log_id": "log_a",
    "entity_text": "bearing",
    "label": "COMPONENT",
}],
                "severity_score": 2,
                "severity_label": "medium",
            },
        ],
    )

    assert result["log_ids_by_window"] == [["log_a", "log_b"], []]
    assert result["window_severity"] == [4.0, None]

    entities = result["entities_by_window"][0]
    assert [entity["log_id"] for entity in entities] == ["log_a", "log_b"]
    assert all(entity["unit_id"] == 1 for entity in entities)
    assert all(entity["cycle"] == 10 for entity in entities)

    severities = result["severity_by_log_by_window"][0]
    assert [item["severity_score"] for item in severities] == [2.0, 4.0]
    assert [item["severity_label"] for item in severities] == [
        "medium", "high"
    ]

    assert result["has_log"].tolist() == [True, False]
    assert torch.count_nonzero(result["text_embeddings"][1]).item() == 0
    assert result["entities_by_window"][1] == []
    assert result["severity_by_log_by_window"][1] == []


def test_missing_severity_does_not_invent_a_window_score():
    result = build_fusion_samples(
        [sensor(1, 10)],
        [log("log_without_severity", 1, 10, 1.0)],
    )

    assert result["log_ids_by_window"] == [["log_without_severity"]]
    assert result["window_severity"] == [None]
    assert (
        result["severity_by_log_by_window"][0][0]["severity_score"]
        is None
    )

def test_fusion_samples_feed_multimodal_model():
    samples = build_fusion_samples(
        [sensor(1, 10), sensor(2, 20)],
        [log("log-1", 1, 10, 2.0)],
    )

    model = MultimodalFusionModel(use_has_log=False)
    model.eval()

    with torch.no_grad():
        predictions = model(
            samples["sensor_embeddings"],
            samples["text_embeddings"],
        )

    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()
    assert samples["has_log"].tolist() == [True, False]
