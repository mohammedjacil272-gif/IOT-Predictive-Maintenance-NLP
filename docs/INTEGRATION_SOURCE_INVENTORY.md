# Integration source inventory

Evidence date: 2026-11-09. This inventory describes the current `integration/jacil-full-project` worktree at `a05509320804f63d9f650c4b1d36b61c7c9e7bca`.

| Area / files | Purpose | Status | Owner | Dependency | Outstanding issue |
|---|---|---|---|---|---|
| `app/`, `tests/test_api.py` | FastAPI local API and route contracts | Implemented; regression-tested | Jacil | Python runtime, frozen artifacts | No live model gateway or external-service verification |
| `assistant/`, `tests/test_assistant_backend.py` | Session, storage, confirmation, attachment, health foundations | Implemented; regression-tested | Jacil/Wahid | SQLite under configured data root | Conversational model gateway and retrieval are not implemented |
| `frontend/`, `frontend/package-lock.json` | React/Vite desktop UI | Build passes | Jacil | Node/npm | Native visual automation not repeated in this cycle |
| `src-tauri/`, `packaging/` | Tauri 2 shell and Windows packaging | Cargo check passes; historical native evidence exists | Jacil | Rust/Cargo, Windows toolchain | Full installer rebuild not repeated |
| `iot/` | Sensor schema, replay, MQTT publisher, Influx ingestion, LSTM inference | Unit/mock coverage passes | Jacil | Optional paho/influx clients and authorized services | No live broker/database configured |
| `nlp/`, `training/nlp_artifact_validation.py` | Log normalization, NER, severity, embedding/artifact validation | Synthetic fixture validation passes | Rafaz | Optional Transformers and approved model assets | Fixture is not genuine paired data; immutable model revision absent |
| `training/`, `models/`, `context/` | Fusion contract, adapters, model and policy checks | Interface/synthetic checks pass | Wahid/Rafaz | Approved paired data and policy decisions | Genuine training is blocked |
| `artifacts/models/fd001_lstm_baseline/`, `artifacts/data/...`, `artifacts/embeddings/...` | Frozen FD001 baseline | Six required hashes match | Jacil | None for read-only verification | Must remain immutable |
| `experiments/azure_pdm/` | Simulated Azure PdM benchmark experiment | Isolated fixture/benchmark code | Rafaz/Wahid | Authorized acquisition root for real runs | No project-authorized acquisition manifest/evidence in this worktree |
| `docs/`, `context/`, `tests/` | Contracts, readiness evidence, regression suite | Reviewed and updated by this handover | Team | Current repository evidence | Historical reports have differing test counts; current commands take precedence |

The Rafaz ZIP was located at `C:\Users\moham\Desktop\IOT_\rafaz_nlp_fusion_readiness_docs.zip` and inspected read-only. It contains six review/setup documents; it does not contain the five Azure CSVs, an acquisition manifest, hashes, licence evidence, or genuine narratives.
