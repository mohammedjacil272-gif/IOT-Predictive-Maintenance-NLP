from __future__ import annotations

from pathlib import Path
from urllib import error

from fastapi.testclient import TestClient
import pytest

from app import main
from assistant.gateway import ModelGatewayConfig, ModelGatewayError, OllamaGateway, bounded_context
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


def test_ollama_gateway_reports_missing_model_from_chat_error():
    class MissingModel:
        def __call__(self, *_args, **_kwargs):
            raise error.HTTPError("http://127.0.0.1:11434/api/chat", 404, "not found", {}, None)

    gateway = OllamaGateway(ModelGatewayConfig(model="qwen2.5:3b", retries=0), opener=MissingModel())
    with pytest.raises(ModelGatewayError, match="model 'qwen2.5:3b' is unavailable"):
        gateway.generate([{"role": "user", "content": "hello"}])


def test_ollama_gateway_distinguishes_connection_refused():
    def refused(*_args, **_kwargs):
        raise error.URLError(ConnectionRefusedError("connection refused"))

    gateway = OllamaGateway(ModelGatewayConfig(model="local-model", retries=0), opener=refused)
    with pytest.raises(ModelGatewayError, match="Cannot connect to local Ollama"):
        gateway.generate([{"role": "user", "content": "hello"}])


def test_ollama_gateway_distinguishes_timeout():
    def timed_out(*_args, **_kwargs):
        raise TimeoutError("timed out")

    gateway = OllamaGateway(ModelGatewayConfig(model="local-model", retries=0), opener=timed_out)
    with pytest.raises(ModelGatewayError, match="timed out after"):
        gateway.generate([{"role": "user", "content": "hello"}])


def test_ollama_gateway_distinguishes_invalid_json():
    class InvalidResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b"not-json"

    gateway = OllamaGateway(ModelGatewayConfig(model="local-model", retries=0), opener=lambda *_args, **_kwargs: InvalidResponse())
    with pytest.raises(ModelGatewayError, match="invalid JSON"):
        gateway.generate([{"role": "user", "content": "hello"}])


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
