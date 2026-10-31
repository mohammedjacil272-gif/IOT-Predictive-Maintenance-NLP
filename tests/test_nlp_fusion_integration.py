"""Architecture-level synthetic NLP package integration tests."""

import json

import pytest

import training.nlp_fusion_pipeline as pipeline
from tests.test_nlp_fusion_pipeline import _sensor_records, _write_package


def test_validated_package_uses_adapter_and_preserves_fusion_contract(
    tmp_path, monkeypatch
):
    _write_package(tmp_path)
    calls = []
    original_adapter = pipeline.adapt_nlp_embedding_metadata

    def recording_adapter(*args, **kwargs):
        calls.append((args, kwargs))
        return original_adapter(*args, **kwargs)

    monkeypatch.setattr(pipeline, "adapt_nlp_embedding_metadata", recording_adapter)
    result = pipeline.build_validated_fusion_samples(
        tmp_path,
        _sensor_records(),
        missing_log_policy="error",
        multiple_log_policy="error",
    )

    assert len(calls) == 1
    assert result["validation"]["status"] == "PASS"
    assert result["validation"]["genuine_data_claim"] is False
    assert result["namespace"] == "synthetic_fixture"
    assert result["samples"]["sensor_embeddings"].shape == (1, 64)
    assert result["samples"]["text_embeddings"].shape == (1, 768)
    assert result["samples"]["source_ids_by_window"][0][0]["log_id"] == "log-0"
    assert result["samples"]["entities_by_window"][0][0]["log_id"] == "log-0"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update(embedding_shape=[1, 767]), "validation did not PASS"),
        (lambda report: report.update(dataset_type="genuine"), "validation did not PASS"),
    ],
)
def test_invalid_package_cannot_bypass_validation(tmp_path, monkeypatch, mutation, message):
    _write_package(tmp_path)
    report_path = tmp_path / "verification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mutation(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("adapter must not receive a failed package")

    monkeypatch.setattr(pipeline, "adapt_nlp_embedding_metadata", fail_if_called)
    with pytest.raises(ValueError, match=message):
        pipeline.build_validated_fusion_samples(
            tmp_path,
            _sensor_records(),
            missing_log_policy="error",
            multiple_log_policy="error",
        )
