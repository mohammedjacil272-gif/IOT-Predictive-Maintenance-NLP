import pytest

from nlp.technician_logs import GenuineLogValidationConfig, validate_genuine_logs


def _record(log_id="b", cycle="4", text=" Bearing wear "):
    return {
        "log_id": log_id,
        "source_system": "cmms",
        "source_machine_id": "machine-7",
        "canonical_unit_id": "7",
        "cycle": cycle,
        "event_timestamp": "2026-01-01T10:00:00+00:00",
        "created_at": "2026-01-01T10:01:00+00:00",
        "ingested_at": "2026-01-01T10:02:00+00:00",
        "available_at": "2026-01-01T10:03:00+00:00",
        "text": text,
        "event_type": "inspection",
        "source_record_id": f"source-{log_id}",
        "source_dataset": "owner-export",
        "source_dataset_version": "2026-01",
        "provenance_reference": "ticket-1",
        "is_synthetic": "false",
        "approval_status": "approved",
    }


def _config():
    return GenuineLogValidationConfig("owner-export", "2026-01")


def test_genuine_intake_is_sorted_and_preserves_multiple_logs():
    prepared, report = validate_genuine_logs(
        [_record("z"), _record("a"), _record("m", text="")],
        _config(),
    )

    assert [row["log_id"] for row in prepared] == ["a", "m", "z"]
    assert report["dropped_rows"] == 0
    assert report["multiple_log_windows"] == 1
    assert report["empty_text_rows"] == 1
    assert prepared[1]["normalized_text"] == ""


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("event_timestamp", "2026-01-01T10:00:00", "timezone"),
        ("canonical_unit_id", "unknown", "canonical_unit_id"),
        ("is_synthetic", "true", "is_synthetic"),
        ("approval_status", "pending", "approval_status"),
    ],
)
def test_genuine_intake_rejects_unapproved_or_ambiguous_values(field, value, match):
    row = _record()
    row[field] = value
    with pytest.raises(ValueError, match=match):
        validate_genuine_logs([row], _config())


def test_genuine_intake_rejects_duplicate_source_identity():
    first = _record("a")
    second = _record("b")
    second["source_record_id"] = first["source_record_id"]

    with pytest.raises(ValueError, match="duplicate source record"):
        validate_genuine_logs([first, second], _config())
