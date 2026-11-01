
"""Rule-based named entity recognition for maintenance logs."""

import re
from typing import Any, Dict, List


# Extend these patterns as you learn the vocabulary in your dataset.
ENTITY_RULES = {
    "COMPONENT": [
        "engine",
        "bearing",
        "fan",
        "compressor",
        "turbine",
        "pump",
        "valve",
        "sensor",
        "shaft",
        "filter",
    ],
    "FAULT": [
        "wear",
        "crack",
        "leak",
        "corrosion",
        "damage",
        "failure",
        "blockage",
        "overheating",
    ],
    "SYMPTOM": [
        "vibration",
        "noise",
        "smoke",
        "overheating",
        "pressure drop",
        "temperature rise",
        "fluid leak",
    ],
    "ACTION": [
        "inspect",
        "replace",
        "repair",
        "clean",
        "lubricate",
        "monitor",
        "tighten",
        "calibrate",
        "shutdown",
        "shut down",
    ],
    "SEVERITY_TERM": [
        "critical",
        "severe",
        "high",
        "moderate",
        "medium",
        "low",
        "minor",
    ],
}

# Matches simple numeric measurements, such as 95°C, 42 PSI or 12.5 bar.
MEASUREMENT_PATTERN = re.compile(
    r"(?<![\w+-])(?<![+-]\s)-?\d+(?:\.\d+)?\s*"
    r"(?:°\s*[CF]|psi|bar|kpa|mpa|rpm|mm|cm|m|"
    r"kg|g|%|pa)\b",
    re.IGNORECASE,
)


def extract_entities(
    text: str,
    log_id: Any = None,
    rule_source: str = "maintenance_rules_v1",
) -> List[Dict[str, Any]]:
    """Extract entities and offsets from the original text.

    Offsets use Python string indices: start is inclusive, end is exclusive.
    Entity text is always sliced from the original input.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    entities = []

    # Rule-based phrase matching, ignoring case.
    for label, terms in ENTITY_RULES.items():
        for term in terms:
            pattern = re.compile(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                re.IGNORECASE,
            )

            for match in pattern.finditer(text):
                entities.append({
                    "log_id": log_id,
                    "entity_text": text[match.start():match.end()],
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "rule_source_id": f"{rule_source}:{label}:{term}",
                })

    # Measurements are extracted separately.
    for match in MEASUREMENT_PATTERN.finditer(text):
        entities.append({
            "log_id": log_id,
            "entity_text": text[match.start():match.end()],
            "label": "MEASUREMENT",
            "start": match.start(),
            "end": match.end(),
            "rule_source_id": f"{rule_source}:MEASUREMENT",
        })

    # Keep output deterministic and in source-text order.
    entities.sort(key=lambda item: (
        item["start"],
        item["end"],
        item["label"],
        item["rule_source_id"],
    ))

    return entities
