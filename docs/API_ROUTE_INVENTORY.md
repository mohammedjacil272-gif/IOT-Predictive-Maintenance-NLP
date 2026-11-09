# API route inventory

All routes are implemented in `app/main.py`. When `IOT_API_TOKEN` is set, every `/api` route except health requires `Authorization: Bearer <token>`.

| Method / route | Request | Response / behavior | Consumer / tests |
|---|---|---|---|
| `GET /api/health` | None | `{status,database_mode,version}` | Shell probe; API tests |
| `GET /api/status` | None | Sensor/NLP/fusion/IoT status | Frontend settings/dashboard; API tests |
| `GET /api/models` | None | Modes and model availability | Frontend; API tests |
| `GET /api/fusion/readiness` | None | Condition map; always fail-closed without approved fusion | Frontend; API tests |
| `POST /api/sensor/infer` | `{records: list[object]}` | Sensor-only prediction metadata; 422 validation, 503 missing artifacts | Frontend; LSTM/API tests |
| `POST /api/technician-log/analyze` | `TechnicianLogRequest` (`log_id`, `text`, optional unit/time/provenance) | Rule-based record/entities/severity and embedding diagnostics | Frontend; NLP/API tests |
| `POST /api/fusion/infer` | Optional `sensor_embedding` (64 finite floats), `text_embedding` (768 finite floats), and identity fields | Validates supplied dimensions/finite values, then remains explicitly unavailable with `prediction:null` | Frontend readiness; API tests |
| `POST /api/infer` | `{mode: sensor_only\|nlp_only\|multimodal_fusion,...}` | Delegates to mode contract; unknown mode 422 | API tests |
| `GET /api/history` | None | Local successful sensor/NLP records | Frontend; API tests |
| `DELETE /api/history` | None | `{cleared:true,records:[]}` | Frontend; API tests |
| `GET /api/diagnostics` | None | Non-secret runtime/path diagnostics | Frontend; desktop tests |
| `GET /api/assistant/model-status` | None | Ollama configuration/installed-model status; no download | Settings/Wahid integration |
| `GET /api/assistant/voice-status` | None | Typed fallback and unavailable audio/TTS status | Voice UI; regression tests |
| `POST /api/assistant/transcribe` | Typed text/content metadata | Typed-input fallback or unavailable provider | Frontend; API tests |
| `POST /api/assistant/speak` | `{text}` | Explicit unavailable TTS response | Voice UI; regression tests |
| `POST /api/assistant/analyze` | `{text,session_id?,machine_id?,...}` | Intent, NLP analysis, confirmation payload | Frontend; API tests |
| `GET /api/machines` | None | `{machines:[]}` | Frontend; storage tests |
| `POST /api/machines` | `MachineRequest` | 201 machine record; 409 duplicate | Frontend; API tests |
| `GET /api/machines/{machine_id}` | Path ID | Machine or 404 | Frontend/storage tests |
| `PATCH /api/machines/{machine_id}` | JSON patch; immutable ID | Updated machine or 404/422 | Storage tests |
| `GET /api/machines/{machine_id}/health` | Path ID | Evidence-backed health or unavailable state | Frontend; API tests |
| `POST /api/assistant/sessions` | `{machine_id}` | 201 session; 404 unknown machine | Frontend; API tests |
| `GET /api/assistant/sessions/{id}` | Path ID | Session or 404 | Frontend/storage tests |
| `GET /api/assistant/sessions/{id}/messages` | Path ID | `{messages:[]}` | Frontend; API tests |
| `POST /api/assistant/sessions/{id}/messages` | `{role,text}` | Stored message; 404/422 on invalid input | Backend contract tests |
| `POST /api/assistant/sessions/{id}/analyze` | `{text,...}` | NLP interpretation, observation and pending confirmation IDs | Frontend; API tests |
| `POST /api/assistant/sessions/{id}/chat` | `{text,include_retrieval?}` | Ollama response or explicit model-unavailable result; bounded context | Wahid integration; mocked gateway tests |
| `POST /api/assistant/sessions/{id}/cancel` | None | Cancelled session; 404/409 invalid transition | Storage tests |
| `GET /api/assistant/confirmations/{id}` | Path ID | Confirmation or 404 | Frontend/backend tests |
| `POST /api/assistant/confirmations/{id}/{confirm\|reject\|cancel}` | Optional decision body | Idempotent decision; 404/409/422 invalid action | Frontend; API tests |
| `POST /api/attachments` | Session, filename, MIME, base64 data | 201 metadata; 404/422 invalid or unsafe upload | Frontend/storage tests |
| `GET /api/attachments/{file_id}` | Path ID | Attachment file or 404/410 | Backend contract tests |
| `POST /api/knowledge/ingest` | Approved local `source_path` | Indexed Markdown/text metadata | Retrieval tests |
| `POST /api/knowledge/search` | `{query,limit?}` | Ranked chunks with source/line references | Retrieval tests |
| `DELETE /api/knowledge/source` | Approved local `source_path` | Removed/not-found status | Retrieval tests |

The conversational model gateway and local retrieval are implemented behind the assistant routes. The API has no MQTT status or InfluxDB proxy route; IoT modules are library/CLI integrations and are reported as `not_checked` by the API until a safe live verification is authorized.
