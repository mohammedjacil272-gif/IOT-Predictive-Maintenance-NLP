from types import SimpleNamespace

import pytest
import torch

from nlp.embeddings import EMBEDDING_DIM, ModelUnavailableError, TextEmbedder
from nlp.ner import extract_entities
from nlp.preprocessing import MAX_TEXT_CHARS, normalize_text, preprocess_log
from nlp.service import InvalidMaintenanceLogError, MaintenanceNLPService
from nlp.severity import score_severity


def test_preprocessing_preserves_source_and_normalizes_unicode_whitespace():
    source = "  PUMP\tpressure  95\N{DEGREE SIGN}C at 42 PSI  "
    row = preprocess_log({"log_id": "L1", "text": source})
    assert row["original_text"] == source
    assert row["normalized_text"] == "pump pressure 95\N{DEGREE SIGN}c at 42 psi"
    assert row["text_status"] == "ok"


@pytest.mark.parametrize("text,status", [(None, "invalid_text"), ("  ", "empty_text"),
                                          ("x" * (MAX_TEXT_CHARS + 1), "text_too_long")])
def test_preprocessing_reports_invalid_empty_and_oversized_text(text, status):
    row = preprocess_log({"log_id": "L1", "text": text})
    assert row["text_status"] == status
    assert row["normalized_text"] == ""


def test_normalize_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_text(None)
    with pytest.raises(ValueError, match="log_id"):
        preprocess_log({"log_id": "  ", "text": "note"})


def test_entity_offsets_slice_original_measurements_and_retain_percent():
    text = "Replace pump; temperature reached 95\N{DEGREE SIGN}C and load was 80%."
    entities = extract_entities(text, log_id="L1")
    for entity in entities:
        assert text[entity["start"]:entity["end"]] == entity["entity_text"]
    measurements = [item["entity_text"] for item in entities if item["label"] == "MEASUREMENT"]
    assert measurements == ["95\N{DEGREE SIGN}C", "80%"]


@pytest.mark.parametrize("text,score,label", [
    ("routine inspection completed", 0, "unknown"),
    ("minor wear observed", 1, "low"),
    ("moderate vibration", 2, "medium"),
    ("bearing failure and abnormal pressure", 3, "high"),
    ("emergency shutdown", 4, "critical"),
])
def test_severity_rules_and_boundaries(text, score, label):
    result = score_severity(text)
    assert result["severity_score"] == score
    assert result["severity_label"] == label


def test_negated_indicator_is_ignored_without_masking_later_clause():
    result = score_severity("No critical failure; high vibration observed")
    assert result["severity_label"] == "high"
    assert [item["term"] for item in result["matched_terms"]] == ["high"]


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 128
        return {"input_ids": torch.ones((1, 3), dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0]])}


class _Model:
    def __call__(self, **inputs):
        values = torch.zeros((1, 3, EMBEDDING_DIM))
        values[:, 0, :] = 1
        values[:, 1, :] = 3
        values[:, 2, :] = 1000
        return SimpleNamespace(last_hidden_state=values)


def test_embedder_masks_padding_and_returns_finite_768_vector():
    embedder = TextEmbedder(tokenizer=_Tokenizer(), model=_Model())
    result = embedder.encode(["note"])
    assert result.shape == (1, 768)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.full((1, 768), 2.0))
    assert embedder.encode([]).shape == (0, 768)
    assert torch.equal(embedder.encode([""])[0], torch.zeros(768))


def test_embedder_rejects_invalid_values_and_empty_or_oversized_batches():
    embedder = TextEmbedder(tokenizer=_Tokenizer(), model=_Model())
    with pytest.raises(TypeError):
        embedder.encode([None])
    with pytest.raises(TypeError):
        embedder.encode("single string")
    with pytest.raises(ValueError, match="characters"):
        embedder.encode(["x" * (MAX_TEXT_CHARS + 1)])


def test_model_load_is_offline_first_and_surfaces_missing_weights(monkeypatch, tmp_path):
    from nlp import embeddings

    embeddings._load_model.cache_clear()
    calls = []

    def unavailable(name, **kwargs):
        calls.append((name, kwargs))
        raise OSError("not cached")

    monkeypatch.setattr(embeddings.AutoTokenizer, "from_pretrained", unavailable)
    monkeypatch.setenv("IOT_NLP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("IOT_NLP_ALLOW_DOWNLOAD", raising=False)
    embedder = TextEmbedder()
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["cache_dir"] == str(tmp_path)
    with pytest.raises(ModelUnavailableError):
        embedder.encode(["maintenance note"])
    embeddings._load_model.cache_clear()


def test_csv_reader_rejects_malformed_rows_and_blank_identifiers(tmp_path):
    from scripts.build_nlp_artifacts import read_logs

    path = tmp_path / "logs.csv"
    path.write_text("log_id,text\nL1,valid note,extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed CSV record"):
        read_logs(path)
    path.write_text("log_id,text\n   ,valid note\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty log_id"):
        read_logs(path)


class _UnavailableEmbedder:
    def encode(self, texts):
        raise ModelUnavailableError("local weights unavailable")


def test_service_returns_clear_rule_based_fallback_and_validates_text():
    service = MaintenanceNLPService(embedder=_UnavailableEmbedder())
    result = service.analyze({"log_id": "L1", "text": "Minor pump leak"})
    assert result["embedding"] is None
    assert result["embedding_status"] == "rule_based_fallback"
    assert result["severity"]["severity_label"] == "high"
    with pytest.raises(InvalidMaintenanceLogError):
        service.analyze({"log_id": "L2", "text": " "})
    with pytest.raises(ValueError):
        service.analyze({"text": "missing identifier"})


class _AvailableEmbedder:
    def encode(self, texts):
        return torch.ones((len(texts), EMBEDDING_DIM))


def test_service_returns_transformer_embedding_contract():
    result = MaintenanceNLPService(embedder=_AvailableEmbedder()).analyze(
        {"log_id": "L2", "text": "Pump inspected"}
    )
    assert result["embedding_status"] == "transformer"
    assert result["embedding_model"] == "distilbert-base-uncased"
    assert result["embedding_dimension"] == 768
    assert len(result["embedding"]) == 768


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_service_rejects_non_finite_embedder_values(value):
    class NonFiniteEmbedder:
        def encode(self, texts):
            vector = torch.ones((len(texts), EMBEDDING_DIM))
            vector[0, 17] = value
            return vector

    with pytest.raises(ValueError, match="non-finite"):
        MaintenanceNLPService(embedder=NonFiniteEmbedder()).analyze(
            {"log_id": "L3", "text": "Pump inspected"}
        )


@pytest.mark.parametrize("output", [None, [], torch.ones((1, EMBEDDING_DIM - 1))])
def test_service_rejects_invalid_embedder_output(output):
    class InvalidEmbedder:
        def encode(self, texts):
            return output

    with pytest.raises(ValueError, match="Embedding service returned"):
        MaintenanceNLPService(embedder=InvalidEmbedder()).analyze(
            {"log_id": "L4", "text": "Pump inspected"}
        )
