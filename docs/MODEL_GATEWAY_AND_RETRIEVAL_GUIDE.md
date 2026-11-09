# Model gateway and retrieval guide

## Local model gateway

The backend has a provider-independent gateway in `assistant/gateway.py` and a local Ollama adapter. It never downloads models. Configure only an already-installed local model:

```powershell
$env:IOT_ASSISTANT_MODEL_PROVIDER = "ollama"
$env:IOT_ASSISTANT_MODEL = "<installed-ollama-model-name>"
$env:IOT_ASSISTANT_MODEL_BASE_URL = "http://127.0.0.1:11434"
$env:IOT_ASSISTANT_MODEL_TIMEOUT = "20"
$env:IOT_ASSISTANT_MODEL_RETRIES = "1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/api/assistant/model-status
```

Use `POST /api/assistant/sessions/{session_id}/chat` only after creating a registered-machine session. Context is bounded to recent messages and retrieval references. Model-unavailable responses are explicit and are not persisted as fabricated assistant answers. The installed local models observed during handover were `qwen2.5:3b` and `qwen2.5-coder:1.5b`; `qwen2.5:3b` returned a local test response through the gateway. No model download was performed.

## Local retrieval

`assistant/retrieval.py` indexes only approved Markdown/text documents under the configured `docs/` root, stores a local index under `.knowledge`, preserves source path and line references, rejects private/database files, and uses deterministic lexical retrieval. This is a safe testable foundation, not a neural embedding claim. PDF parsing and a production embedding provider remain Wahid-owned follow-up work.
