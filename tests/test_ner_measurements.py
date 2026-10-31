import pytest

from nlp.ner import extract_entities


def test_unicode_celsius_measurement_preserves_text_offsets_and_rule_id():
    text = "Bearing temperature reached 95\u00b0C."
    start = text.index("95\u00b0C")

    measurements = [
        entity for entity in extract_entities(text, log_id="L-temp")
        if entity["label"] == "MEASUREMENT"
    ]

    assert measurements == [{
        "log_id": "L-temp",
        "entity_text": "95\u00b0C",
        "label": "MEASUREMENT",
        "start": start,
        "end": start + len("95\u00b0C"),
        "rule_source_id": "maintenance_rules_v1:MEASUREMENT",
    }]


def test_celsius_measurement_accepts_whitespace_before_unit():
    text = "Temperature was 95 \u00b0C during inspection."
    start = text.index("95 \u00b0C")

    measurements = [
        entity for entity in extract_entities(text)
        if entity["label"] == "MEASUREMENT"
    ]

    assert len(measurements) == 1
    assert measurements[0]["entity_text"] == "95 \u00b0C"
    assert (measurements[0]["start"], measurements[0]["end"]) == (
        start, start + len("95 \u00b0C")
    )


@pytest.mark.parametrize(
    "measurement",
    ["95\u00b0C", "95 \u00b0C", "-5\u00b0C", "36.5\u00b0C", "-2.5\u00b0C"],
)
def test_celsius_measurements_preserve_exact_text_offsets_and_provenance(measurement):
    text = f"Temperature: {measurement}."
    start = text.index(measurement)

    entities = [
        entity for entity in extract_entities(text, log_id="fixture-log")
        if entity["label"] == "MEASUREMENT"
    ]

    assert entities == [{
        "log_id": "fixture-log",
        "entity_text": measurement,
        "label": "MEASUREMENT",
        "start": start,
        "end": start + len(measurement),
        "rule_source_id": "maintenance_rules_v1:MEASUREMENT",
    }]
