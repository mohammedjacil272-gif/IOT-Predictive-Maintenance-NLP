import pytest

from nlp.embeddings import embedding_diagnostics


def test_embedding_diagnostics_is_offline_and_structured(tmp_path):
    result = embedding_diagnostics(tmp_path)
    assert result["network_download_attempted"] is False
    assert result["ready"] is False
    assert result["model_name"] == "distilbert-base-uncased"


def test_embedder_reports_optional_dependency_or_local_artifact_failure():
    from nlp.embeddings import TextEmbedder

    with pytest.raises(RuntimeError, match="NLP embeddings unavailable"):
        TextEmbedder()
