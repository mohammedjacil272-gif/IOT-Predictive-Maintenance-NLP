
"""Heuristic severity scoring for synthetic maintenance logs.

Scores are rule-based estimates, not validated real-world labels.
"""

import re
from typing import Any, Dict


# Higher score wins when multiple positive severity terms occur.
SEVERITY_RULES = {
    1: ["low", "minor", "mild"],
    2: ["medium", "moderate", "noticeable"],
    3: ["high", "severe", "major", "serious"],
    4: ["critical", "catastrophic", "emergency"],
}

# These patterns indicate that a severity phrase may be negated.
NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|neither|never|absence of|"
    r"not detected|not observed)\b",
    re.IGNORECASE,
)

def score_severity(text: str) -> Dict[str, Any]:
    """Return a severity score from 0 to 4 and supporting evidence."""
    if not isinstance(text, str) or not text.strip():
        return {
            "severity_score": 0,
            "severity_label": "unknown",
            "matched_terms": [],
            "explanation": "Empty or invalid text",
        }

    matches = []

    for score, terms in SEVERITY_RULES.items():
        for term in terms:
            pattern = re.compile(
                r"\b" + re.escape(term) + r"\b",
                re.IGNORECASE,
            )

            for match in pattern.finditer(text):
                # Examine a small context window before the term.
                preceding = text[max(0, match.start() - 35):match.start()]

                # Ignore terms directly preceded by a negation cue.
                if NEGATION_PATTERN.search(preceding):
                    continue

                matches.append({
                    "term": match.group(),
                    "score": score,
                    "start": match.start(),
                    "end": match.end(),
                })

    if not matches:
        return {
            "severity_score": 0,
            "severity_label": "unknown",
            "matched_terms": [],
            "explanation": "No unnegated severity term matched",
        }

    # Explicit policy: choose the highest remaining severity.
    highest_score = max(item["score"] for item in matches)

    labels = {
        0: "unknown",
        1: "low",
        2: "medium",
        3: "high",
        4: "critical",
    }

    return {
        "severity_score": highest_score,
        "severity_label": labels[highest_score],
        "matched_terms": matches,
        "explanation": "Highest unnegated rule-based severity term",
    }
