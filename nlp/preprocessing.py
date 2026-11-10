
"""Validate and normalize technician maintenance logs.

The original text is preserved so that NER character offsets can be
checked against the exact source text.
"""

from typing import Any, Dict
import re
import unicodedata

MAX_TEXT_CHARS = 20_000


def normalize_text(text: str) -> str:
    """Normalize text without changing the original source value."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def preprocess_log(log: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one log and return its original and normalized text."""
    if not isinstance(log, dict):
        raise TypeError("Each log must be a dictionary.")

    log_id = log.get("log_id")
    if (not isinstance(log_id, (str, int)) or isinstance(log_id, bool)
            or not str(log_id).strip()):
        raise ValueError("Missing required field: log_id")

    original = log.get("text")

    if not isinstance(original, str):
        status = "invalid_text"
        normalized = ""
    elif not original.strip():
        status = "empty_text"
        normalized = ""
    elif len(original) > MAX_TEXT_CHARS:
        status = "text_too_long"
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
