# AI assistant connection verification

Refresh 2026-11-09: local retrieval and bounded model context were reverified; the retrieval index remains lexical and approved-root limited, and PDF/neural embedding providers remain unconfigured.

Evidence date: 2026-11-09.

| Connection | Command / evidence | Result | Status |
|---|---|---|---|
| Frontend → FastAPI | `cd frontend; npm run build`; `tests/test_api.py` | Build and local API contracts pass | VERIFIED LOCAL |
| Assistant sessions/history | `python -m pytest tests/test_api.py tests/test_assistant_backend.py tests/test_storage_hardening.py -q` | Covered by full suite; session restore/storage lifecycle passes | VERIFIED LOCAL |
| Confirmation workflow | `tests/test_api.py` confirmation lifecycle | Confirm/reject/cancel and duplicate behavior covered | VERIFIED LOCAL |
| Attachments | `tests/test_api.py`, `tests/test_storage_hardening.py` | Validation and local storage covered | VERIFIED LOCAL |
| Technician-log NLP | `POST /api/technician-log/analyze`; NLP tests | Structured rule-based output and provenance pass | VERIFIED LOCAL / SYNTHETIC |
| Sensor inference | `tests/test_lstm_inference.py`, `tests/test_api.py` | Frozen checkpoint/scaler adapter and validation pass | VERIFIED LOCAL |
| Fusion readiness | `GET /api/fusion/readiness`, fusion tests | Fails closed with paired-data/policy blockers | VERIFIED LOCAL |
| Model gateway | `GET /api/assistant/model-status`, mocked gateway tests, local `qwen2.5:3b` response | Existing local model responded; cloud providers remain absent | VERIFIED LOCAL / LIVE OLLAMA |
| Retrieval | `/api/knowledge/*`, synthetic Markdown index/search tests | Lexical source-grounded local index works; neural/PDF providers absent | VERIFIED LOCAL / PARTIAL |
| MQTT | Unit/mock tests only; no authorized broker endpoint/token | No live publish/subscribe attempted | CONFIGURED BUT NOT VERIFIED |
| InfluxDB | Unit/mock tests only; no authorized service/token | No live write/query attempted | CONFIGURED BUT NOT VERIFIED |
| Tauri sidecar | Existing desktop report; current `cargo check` | Historical smoke evidence; not repeated end-to-end here | HISTORICAL + BUILD VERIFIED |

No credentials or tokens were printed. No production topic, bucket, database, user history, or protected artifact was modified. Live-service verification requires Jacil to provide an isolated test broker/bucket, credentials, topic/measurement, retention/cleanup instructions, and authorization.
