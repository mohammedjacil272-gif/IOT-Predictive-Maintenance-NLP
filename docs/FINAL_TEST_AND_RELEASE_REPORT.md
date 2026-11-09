# Final test and release report

Refresh 2026-11-09: the final suite result is 397 passed, 3 skipped, 1 warning. The fusion API now validates 64-D/768-D finite inputs while remaining fail-closed without approved paired data/checkpoint. Frontend audit was run from frontend with the existing lockfile and reported 0 vulnerabilities.

Evidence date: 2026-11-09. No commit or push was made.

| Check | Command | Result |
|---|---|---|
| Full Python suite | `python -m pytest -q` | 396 passed, 3 skipped, 1 warning, 25 subtests |
| Assistant/gateway/retrieval/API targeted | `python -m pytest tests/test_assistant_gateway_retrieval.py tests/test_api.py tests/test_assistant_backend.py -q` | 21 passed after gateway/retrieval changes |
| Python compilation | `python -m compileall -q app assistant iot nlp training models` | Passed |
| FD001 verifier | `python scripts/verify_lstm_baseline.py` | Passed; frozen artifacts and contracts valid |
| NLP verifier | `python scripts/verify_nlp.py --package-dir artifacts` | Overall PASS with expected provenance/revision/timestamp/authenticity blockers |
| NLP/fusion compatibility | `python scripts/verify_nlp_fd001_compatibility.py` | Passed synthetic interface checks |
| Fusion provenance | `python scripts/verify_fusion_provenance.py` | Passed synthetic provenance checks |
| Local Ollama gateway | `IOT_ASSISTANT_MODEL=qwen2.5:3b` gateway status + one local response | Passed; installed model responded `READY`; no model download performed |
| Frontend build | `npm run build` from `frontend/` | Passed |
| Frontend audit | `npm audit --audit-level=moderate` | 0 vulnerabilities |
| Rust check | `cargo check --manifest-path src-tauri/Cargo.toml` | Passed |
| Diff whitespace | `git diff --check` | Passed |
| Global pip check | `python -m pip check` | Fails on unrelated global aider-chat conflicts; no global changes made |

No live MQTT/InfluxDB, installer rebuild, or visual WebView automation was claimed. Ollama was verified locally because the installed `qwen2.5:3b` model was already present. No protected FD001 artifact was changed.
