
"""Validate and normalize technician maintenance logs.

The original text is preserved so that NER character offsets can be
checked against the exact source text.
"""

from typing import Any, Dict
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text without changing the original source value."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def preprocess_log(log: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one log and return its original and normalized text."""
    if not isinstance(log, dict):
        raise TypeError("Each log must be a dictionary.")

    if "log_id" not in log or log["log_id"] is None:
        raise ValueError("Missing required field: log_id")

    original = log.get("text")

    if not isinstance(original, str):
        status = "invalid_text"
        normalized = ""
    elif not original.strip():
        status = "empty_text"
        normalized = ""
    else:
        status = "ok"
        normalized = normalize_text(original)

    result = dict(log)
    result["original_text"] = original
    result["normalized_text"] = normalized
    result["text_status"] = status

    return result


def preprocess_logs(logs):
    """Preprocess multiple logs while preserving their order."""
    return [preprocess_log(log) for log in logs]
