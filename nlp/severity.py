"""Transparent heuristic severity scoring; not a validated risk estimate."""

import re
from typing import Any, Dict

SEVERITY_RULES = {
    1: ["low", "minor", "mild"],
    2: ["medium", "moderate", "noticeable"],
    3: ["high", "severe", "major", "serious", "failure", "failed",
        "overheating", "leak", "fire", "warning", "alarm", "abnormal",
        "pressure drop", "temperature rise"],
    4: ["critical", "catastrophic", "emergency", "shutdown", "shut down",
        "safety hazard", "explosion"],
}
NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|neither|never|absence of|not detected|not observed)\b",
    re.IGNORECASE,
)


def _is_negated(text: str, start: int) -> bool:
    # Limit scope to the current clause, then require a nearby negation cue.
    preceding = text[:start].rsplit(";", 1)[-1].rsplit(".", 1)[-1]
    preceding = re.split(r"\bbut\b", preceding, flags=re.IGNORECASE)[-1]
    cues = list(NEGATION_PATTERN.finditer(preceding))
    if not cues:
        return False
    tail = preceding[cues[-1].end():]
    return len(re.findall(r"\b\w+\b", tail)) <= 4


def score_severity(text: str) -> Dict[str, Any]:
    """Return score, label and exact rule evidence from the note.

    Scores are deterministic: explicit emergency/shutdown terms map to 4;
    severe damage or abnormal operating conditions map to 3; explicit
    moderate terms map to 2; low/minor terms map to 1. Highest match wins.
    A negation within four intervening words in the same clause suppresses a
    match. This small rule set is only a transparent heuristic.
    """
    if not isinstance(text, str) or not text.strip():
        return {"severity_score": 0, "severity_label": "unknown",
                "matched_terms": [], "explanation": "Empty or invalid text"}

    matches = []
    for score, terms in SEVERITY_RULES.items():
        for term in terms:
            pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                if not _is_negated(text, match.start()):
                    matches.append({"term": match.group(), "score": score,
                                    "start": match.start(), "end": match.end()})

    labels = {0: "unknown", 1: "low", 2: "medium", 3: "high", 4: "critical"}
    if not matches:
        return {"severity_score": 0, "severity_label": "unknown",
                "matched_terms": [], "explanation": "No unnegated severity or fault indicator matched"}
    highest = max(item["score"] for item in matches)
    return {"severity_score": highest, "severity_label": labels[highest],
            "matched_terms": sorted(matches, key=lambda item: (item["start"], item["end"])),
            "explanation": "Highest unnegated rule-based severity indicator"}
