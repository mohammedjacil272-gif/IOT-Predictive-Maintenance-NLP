# AI assistant feature matrix

Verification refresh 2026-11-09: conversation context is bounded before gateway calls; local retrieval is implemented for approved Markdown/text roots and source references; the local Ollama gateway was live-verified with an already-installed model. PDF/neural retrieval, voice, and TTS remain optional/unavailable.

| Feature | Frontend | API route(s) | Contract / dependency | Auth | Coverage | Status | Owner / next action |
|---|---|---|---|---|---|---|---|
| Backend health/status | Dashboard/settings | `GET /api/health`, `/api/status`, `/api/diagnostics` | JSON readiness/status | Local bearer token except health | API tests | implemented | Jacil; keep evidence-backed |
| Model availability | Dashboard/settings | `GET /api/models`, `/api/fusion/readiness` | Sensor 30x17; fusion fail-closed | Local bearer token | API tests | implemented | Jacil; extend only with approved models |
| Sensor inference | Sensor view | `POST /api/sensor/infer`, `/api/infer` | Validated records; frozen FD001 model | Local bearer token | API/LSTM tests | implemented | Jacil; live IoT validation pending |
| Technician-log analysis | Logs/assistant | `POST /api/technician-log/analyze`, `/api/assistant/analyze` | Rule-based entities/severity and provenance | Local bearer token | NLP/API tests | implemented | Rafaz; genuine data evidence pending |
| Session creation/restoration | Assistant view | session CRUD routes | SQLite local storage | Local bearer token | assistant API/storage tests | implemented | Wahid may add model context |
| Conversation history | Assistant/history | messages/history routes | Local records; no LLM memory | Local bearer token | API/storage tests | partial | Wahid: define context-window policy |
| Confirmation workflow | Assistant view | confirmation get/decision routes | Confirm/reject/cancel; duplicate-safe | Local bearer token | API tests | implemented | Jacil/Wahid; bind consequential tools later |
| Attachments | Assistant view | `POST /api/attachments`, `GET /api/attachments/{id}` | Allowlisted MIME/size, local storage | Local bearer token | storage/API tests | implemented | Wahid: add approved parsing/retrieval |
| Voice/transcription | Assistant | `POST /api/assistant/transcribe` | Typed fallback only | Local bearer token | API tests | partial | Wahid: choose approved provider or retain fallback |
| Machine registry/health | Machines/dashboard | `/api/machines*` | Explicit machine records and observations | Local bearer token | API/storage tests | implemented | Jacil; no fabricated plant mappings |
| Model gateway/retries/timeouts | Assistant/settings | `GET /api/assistant/model-status`, `POST /api/assistant/sessions/{id}/chat` | Ollama adapter, bounded context, finite retry/timeout; model must already be installed | Local bearer token | Mock gateway/API tests plus local qwen2.5:3b response | implemented locally | Wahid: choose approved default and add richer orchestration |
| Knowledge ingestion/retrieval | Future assistant source panel | `/api/knowledge/*`, chat retrieval | Approved Markdown/text roots; lexical local index with source lines | Local bearer token | Synthetic retrieval tests | partial | Wahid: add approved embedding provider/PDF parser if needed |
| Multimodal reasoning | Readiness only | `POST /api/fusion/infer` | 64-D + 768-D metadata-aligned inputs | Local bearer token | fail-closed tests | blocked | Wahid/Rafaz: await paired data and approvals |
