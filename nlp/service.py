"""Application-facing orchestration for maintenance-log NLP.

This repository has no HTTP backend. The service provides one stable result
contract for a future route or another local application caller.
"""

from typing import Any, Dict

import torch

from nlp.embeddings import EMBEDDING_DIM, MODEL_NAME, ModelUnavailableError, get_text_embedder
from nlp.ner import extract_entities
from nlp.preprocessing import MAX_TEXT_CHARS, preprocess_log
from nlp.severity import score_severity


class InvalidMaintenanceLogError(ValueError):
    """Input cannot be accepted as a maintenance log."""


class MaintenanceNLPService:
    """Analyze one log with deterministic rules and optional local embeddings."""

    def __init__(self, embedder=None):
        self.embedder = embedder

    def analyze(self, log: Dict[str, Any]) -> Dict[str, Any]:
        processed = preprocess_log(log)
        if (not isinstance(processed.get("log_id"), (str, int))
                or isinstance(processed["log_id"], bool)
                or not str(processed["log_id"]).strip()):
            raise InvalidMaintenanceLogError("log_id must be a non-empty string or integer")
        if processed["text_status"] != "ok":
            raise InvalidMaintenanceLogError(
                f"text must be a non-empty string of at most {MAX_TEXT_CHARS} characters "
                f"({processed['text_status']})"
            )

        original = processed["original_text"]
        entities = extract_entities(original, log_id=processed["log_id"])
        severity = score_severity(processed["normalized_text"])
        embedder = self.embedder if self.embedder is not None else get_text_embedder()
        try:
            vector = embedder.encode([processed["normalized_text"]])
            try:
                vector = torch.as_tensor(vector)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError("Embedding service returned invalid output") from exc
            if tuple(vector.shape) != (1, EMBEDDING_DIM):
                raise ValueError("Embedding service returned an unexpected shape")
            if not bool(torch.isfinite(vector).all()):
                raise ValueError("Embedding service returned non-finite values")
            embedding = vector[0].tolist()
            embedding_status = "transformer"
            fallback_reason = None
        except ModelUnavailableError as exc:
            embedding = None
            embedding_status = "rule_based_fallback"
            fallback_reason = str(exc)

        return {
            "log_id": processed["log_id"],
            "original_text": original,
            "normalized_text": processed["normalized_text"],
            "entities": entities,
            "severity": severity,
            "embedding": embedding,
            "embedding_model": MODEL_NAME if embedding is not None else None,
            "embedding_dimension": EMBEDDING_DIM if embedding is not None else None,
            "embedding_status": embedding_status,
            "fallback_reason": fallback_reason,
        }
