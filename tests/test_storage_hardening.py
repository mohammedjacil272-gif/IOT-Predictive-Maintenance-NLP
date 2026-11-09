import json
import threading

import pytest

from assistant.storage import AssistantStorage


def test_foreign_keys_state_transitions_and_restart(tmp_path):
    path = tmp_path / "assistant.sqlite3"
    store = AssistantStorage(path)
    store.save_machine("m1", {"machine_id": "m1", "is_synthetic": True})
    store.save_session("s1", "m1", "IDLE", {})
    store.add_message("s1", "user", "hello")
    store.set_state("s1", "PROCESSING")
    with pytest.raises(ValueError): store.set_state("s1", "LISTENING")
    restarted = AssistantStorage(path)
    assert restarted.get_session("s1")["state"] == "PROCESSING"
    assert restarted.list_messages("s1")[0]["text"] == "hello"
    with pytest.raises(KeyError): restarted.add_message("missing", "user", "x")


def test_json_history_migration_is_backed_up_and_idempotent(tmp_path):
    source = tmp_path / "history.json"
    source.write_text(json.dumps([{"mode": "nlp_only", "model_id": "rule_based"}]), encoding="utf-8")
    store = AssistantStorage(tmp_path / "assistant.sqlite3")
    first = store.migrate_json_history(source)
    second = store.migrate_json_history(source)
    assert first["imported"] == 1
    assert second["imported"] == 0
    assert source.with_suffix(".json.backup").exists()
    assert source.exists()


def test_duplicate_machine_and_invalid_observation_are_rejected(tmp_path):
    store = AssistantStorage(tmp_path / "assistant.sqlite3")
    store.save_machine("m1", {})
    with pytest.raises(ValueError): store.save_machine("m1", {})
    store.save_session("s1", "m1", "IDLE", {})
    with pytest.raises(ValueError): store.save_observation("o1", "s1", "m1", "unknown", {})
    store.save_observation("o1", "s1", "m1", "confirmed", {})
    with pytest.raises(ValueError): store.save_observation("o1", "s1", "m1", "rejected", {})


def test_concurrent_message_writes_are_serialized(tmp_path):
    path = tmp_path / "assistant.sqlite3"
    seed = AssistantStorage(path); seed.save_session("s1", None, "IDLE", {})
    errors = []
    def write(i):
        try: AssistantStorage(path).add_message("s1", "user", str(i))
        except Exception as exc: errors.append(exc)
    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    assert len(AssistantStorage(path).list_messages("s1")) == 8
