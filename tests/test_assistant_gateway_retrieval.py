from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from assistant.gateway import ModelGatewayConfig, OllamaGateway, bounded_context
from assistant.retrieval import LocalKnowledgeIndex, RetrievalError


client = TestClient(main.app)


def test_bounded_context_keeps_recent_messages_and_limits_size():
    messages = [{"role": "user", "text": "old"}, {"role": "assistant", "text": "middle"}, {"role": "user", "text": "new"}]
    result = bounded_context(messages, max_messages=2, max_chars=3)
    assert result == [{"role": "user", "content": "new"}]


def test_ollama_gateway_generates_from_mocked_response():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"message":{"content":"Local answer"}}'

    gateway = OllamaGateway(ModelGatewayConfig(model="local-model", retries=0), opener=lambda *_args, **_kwargs: Response())
    assert gateway.generate([{"role": "user", "content": "hello"}])["text"] == "Local answer"


def test_ollama_gateway_unconfigured_is_explicit():
    gateway = OllamaGateway(ModelGatewayConfig(model=""))
    try:
        gateway.generate([])
    except Exception as exc:
        assert "not configured" in str(exc).lower() or "configured" in str(exc).lower()
    else:
        raise AssertionError("unconfigured gateway must not generate a response")


def test_local_knowledge_index_is_source_grounded_and_excludes_private_files(tmp_path):
    source_root = tmp_path / "approved"
    source_root.mkdir()
    document = source_root / "manual.md"
    document.write_text("Bearing vibration requires inspection.\nUse the approved SOP.", encoding="utf-8")
    index = LocalKnowledgeIndex(tmp_path / "index", allowed_roots=(source_root,))
    result = index.ingest(document)
    assert result["chunks"] == 1
    found = index.search("bearing vibration")
    assert found and found[0]["source_name"] == "manual.md"
    assert found[0]["line_start"] == 1
    private = source_root / ".env"
    private.write_text("TOKEN=secret", encoding="utf-8")
    try:
        index.ingest(private)
    except RetrievalError:
        pass
    else:
        raise AssertionError("private files must be rejected")


def test_chat_route_persists_user_and_mocked_assistant_message(monkeypatch, tmp_path):
    original_storage, original_gateway = main.ASSISTANT_STORAGE, main.MODEL_GATEWAY
    main.ASSISTANT_STORAGE = main.AssistantStorage(tmp_path / "assistant.sqlite3")

    class FakeGateway:
        def generate(self, messages):
            assert any(item["role"] == "user" for item in messages)
            return {"text": "Mocked local response", "provider": "test", "model": "fixture", "response_type": "model"}

        def status(self):
            return {"status": "available", "provider": "test"}

    main.MODEL_GATEWAY = FakeGateway()
    try:
        machine = client.post("/api/machines", json={"machine_id": "gateway-machine", "display_name": "Gateway fixture", "is_synthetic": True})
        assert machine.status_code == 201
        session = client.post("/api/assistant/sessions", json={"machine_id": "gateway-machine"})
        sid = session.json()["session_id"]
        response = client.post(f"/api/assistant/sessions/{sid}/chat", json={"text": "What is the status?", "include_retrieval": False})
        assert response.status_code == 200
        assert response.json()["response"] == "Mocked local response"
        messages = client.get(f"/api/assistant/sessions/{sid}/messages").json()["messages"]
        assert [item["role"] for item in messages] == ["user", "assistant"]
    finally:
        main.ASSISTANT_STORAGE, main.MODEL_GATEWAY = original_storage, original_gateway


def test_voice_routes_are_explicit_about_unavailable_audio():
    assert client.get("/api/assistant/voice-status").json()["input"]["status"] == "typed_fallback"
    response = client.post("/api/assistant/speak", json={"text": "hello"})
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
