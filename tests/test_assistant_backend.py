import json

import pytest

from assistant.attachments import attachment_contract, store_attachment
from assistant.confirmation import confirmation_payload, missing_information
from assistant.health import machine_health
from assistant.intents import classify_intent
from assistant.storage import AssistantStorage
from assistant.voice import TranscriptionRequest, UnavailableTTSProvider, UnavailableTranscriptionProvider


def test_intents_and_missing_information_are_deterministic():
    assert classify_intent("Give me a rundown of machine 12") == "machine_rundown"
    assert "When was this observed?" in missing_information({"machine_id": "12", "component": "bearing"})
    payload = confirmation_payload("bearings are slow", {"machine_id": "12", "component": "bearing"})
    assert payload["original_transcript"] == "bearings are slow"
    assert payload["confirmation_question"] == "Is this what you meant?"


def test_storage_preserves_pending_and_confirmed_observations(tmp_path):
    store = AssistantStorage(tmp_path / "assistant.sqlite3")
    store.save_machine("12", {"machine_id": "12", "display_name": "Fixture machine", "is_synthetic": True})
    store.save_session("s1", "12", "IDLE", {"is_synthetic": True})
    store.save_observation("o1", "s1", "12", "pending", {"interpretation": {"component": "bearing"}})
    store.save_observation("o2", "s1", "12", "confirmed", {"interpretation": {"component": "bearing"}})
    rows = store.list_observations("12")
    assert {row["status"] for row in rows} == {"pending", "confirmed"}
    assert store.get_machine("12")["is_synthetic"] is True


def test_voice_and_attachment_boundaries(tmp_path):
    provider = UnavailableTranscriptionProvider()
    assert provider.transcribe(TranscriptionRequest(typed_text="hello"))["status"] == "typed_input"
    assert provider.transcribe(TranscriptionRequest())["status"] == "unavailable"
    assert UnavailableTTSProvider().synthesize("hello")["status"] == "unavailable"
    meta = store_attachment(b"%PDF-1.7 demo", "../report.pdf", "application/pdf", tmp_path, session_id="s1")
    assert meta["original_filename"] == "report.pdf"
    assert len(meta["sha256"]) == 64
    with pytest.raises(ValueError): store_attachment(b"x", "x.exe", "application/octet-stream", tmp_path, session_id="s1")
    with pytest.raises(ValueError): store_attachment(b"not-pdf", "x.pdf", "application/pdf", tmp_path, session_id="s1")


@pytest.mark.parametrize(
    "filename,mime_type,data",
    [
        ("note.txt", "text/plain", b"bearing vibration"),
        ("rows.csv", "text/csv", b"unit_id,cycle\n1,30\n"),
        ("record.json", "application/json", b'{"unit_id": 1}'),
    ],
)
def test_text_attachment_contract_accepts_supported_formats(tmp_path, filename, mime_type, data):
    metadata = store_attachment(data, filename, mime_type, tmp_path, session_id="s1")
    assert metadata["validation_status"] == "stored_unanalyzed"
    assert (tmp_path / metadata["file_id"]).read_bytes() == data


def test_attachment_contract_rejects_mime_mismatch_empty_malformed_and_traversal_extension(tmp_path):
    assert ".json" in attachment_contract()["formats"]
    with pytest.raises(ValueError, match="invalid"):
        store_attachment(b'{"ok": true}', "record.txt", "application/json", tmp_path, session_id="s1")
    with pytest.raises(ValueError, match="invalid"):
        store_attachment(b"not json", "record.json", "application/json", tmp_path, session_id="s1")
    with pytest.raises(ValueError, match="empty"):
        store_attachment(b"", "empty.txt", "text/plain", tmp_path, session_id="s1")
    metadata = store_attachment(b"safe", "..\\..\\safe.txt", "text/plain", tmp_path, session_id="s1")
    assert metadata["original_filename"] == "safe.txt"
    assert (tmp_path / metadata["file_id"]).resolve().parent == tmp_path.resolve()


def test_health_does_not_convert_rul_to_percentage():
    result = machine_health({"machine_id": "12"}, sensor_prediction={"model_id": "fd001_lstm_baseline", "prediction": 42.0})
    assert result["evidence"][0]["semantics"].startswith("sensor-only")
    assert "health percentage" in result["evidence"][0]["semantics"]
