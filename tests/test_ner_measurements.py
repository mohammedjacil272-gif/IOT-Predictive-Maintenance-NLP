
"""Regression coverage for maintenance measurement extraction."""

import pytest

from nlp.ner import extract_entities


@pytest.mark.parametrize(
    "measurement",
    ["95°C", "95 °C", "-5°C", "-2.5°C", "36.5°C"],
)
def test_celsius_measurements_preserve_text_and_offsets(measurement):
    text = f"Temperature reached {measurement}."
    start = text.index(measurement)

    measurements = [
        entity
        for entity in extract_entities(text, log_id="fixture-log")
        if entity["label"] == "MEASUREMENT"
    ]

    assert measurements == [
        {
            "log_id": "fixture-log",
            "entity_text": measurement,
            "label": "MEASUREMENT",
            "start": start,
            "end": start + len(measurement),
            "rule_source_id": "maintenance_rules_v1:MEASUREMENT",
        }
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Temperature: +5°C",
        "Temperature: + 5°C",
        "Temperature: - 5°C",
        "Temperature: --5°C",
        "Temperature: 5 C",
    ],
)
def test_malformed_celsius_measurements_are_not_extracted(text):
    assert [
        entity
        for entity in extract_entities(text)
        if entity["label"] == "MEASUREMENT"
    ] == []


def test_non_string_measurement_input_is_rejected():
    with pytest.raises(TypeError, match="text must be a string"):
        extract_entities(None)


@pytest.mark.parametrize(
    "measurement",
    ["-5 PSI", "95 PSI", "12.5 bar"],
)
def test_signed_positive_and_decimal_measurement_units_are_supported(
    measurement,
):
    entities = [
        entity
        for entity in extract_entities(f"Reading: {measurement}")
        if entity["label"] == "MEASUREMENT"
    ]

    assert [entity["entity_text"] for entity in entities] == [measurement]


@pytest.mark.parametrize(
    "text",
    ["95 PSIx", "95 C", "- 5 PSI", "+ 5 PSI"],
)
def test_measurement_unit_boundaries_and_malformed_signs_are_rejected(text):
    assert [
        entity
        for entity in extract_entities(text)
        if entity["label"] == "MEASUREMENT"
    ] == []
