# Project status

| Area | Status | Evidence |
|---|---|---|
| FD001 preprocessing/LSTM | COMPLETE AND VERIFIED | Existing tests and validation reports; official test evaluation remains unperformed |
| IoT replay/MQTT/Influx adapters | IMPLEMENTED BUT NOT FULLY VERIFIED | Unit/contract tests; live services not required for app startup |
| NLP schema/rules | COMPLETE AND VERIFIED | Existing NLP tests |
| Transformer embeddings | BLOCKED BY EXTERNAL DEPENDENCY | `transformers` and local DistilBERT weights unavailable; offline failure path validated |
| Fusion alignment/provenance/model | IMPLEMENTED BUT NOT FULLY VERIFIED | Current main tests and synthetic contract checks; no genuine training/result |
| Fusion predictive performance | BLOCKED BY EXTERNAL DEPENDENCY | Approved paired real data, mapping, target/cutoff, and authorization absent |
| Genuine data readiness | BLOCKED BY EXTERNAL DEPENDENCY | No approved technician logs, crosswalk, shared timestamps, or paired observations |
| Backend API | COMPLETE AND VERIFIED | Endpoint, invalid-input, readiness and unavailable-fusion runtime checks |
| UI | IMPLEMENTED BUT NOT FULLY VERIFIED | Static response and JavaScript syntax validated; manual browser click-through remains |
| Unified inference/history | COMPLETE AND VERIFIED | Explicit mode dispatch and local sensor/NLP history tests |
| Technician assistant backend | IMPLEMENTED BUT NOT FULLY VERIFIED | Persistent sessions/messages, validated transitions, confirmation lifecycle, machine CRUD, attachments, health and SQLite migration helper |
| Voice services | BLOCKED BY EXTERNAL DEPENDENCY | Typed-input fallback exists; no local speech/TTS providers installed |
| React/Tauri desktop shell | IMPLEMENTED BUT NOT FULLY VERIFIED | Frontend/Tauri scaffold created; Rust/Cargo/PyInstaller unavailable, no installer built |

No status above implies live MQTT/Influx operation or real-world predictive performance.
