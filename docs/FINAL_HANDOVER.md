# MaintainIQ final handover

Execution refresh: 2026-11-09. No commit, push, branch switch, reset, clean,
or destructive restore was performed.

## Final status

**READY FOR SENSOR-ONLY DEMONSTRATION — MULTIMODAL DATA BLOCKED**

The local sensor-only baseline, rule-based NLP, API, attachment contract, and
safe local IoT replay paths are implemented and tested. Genuine multimodal
evaluation is blocked by the absence of authorized paired sensor/log data,
crosswalk, temporal approvals, pinned transformer assets, and a fusion
checkpoint.

## Phase status

- FD001 artifact handover: locally verified; external artifact distribution
  decision remains open because large checkpoints are Git-ignored.
- NLP pipeline: rule-based path implemented and verified; transformer path
  optional/unavailable without approved local weights.
- Genuine paired alignment: blocked; no authorized paired dataset exists.
- Multimodal fusion: interfaces verified; training/evaluation blocked.
- IoT: local replay, validation, and explicit timestamp policy verified; live
  MQTT/Influx/MongoDB infrastructure not connected.
- Backend: implemented and API-tested.
- Frontend: TypeScript and production Vite build verified; Tauri/installer not
  verified without Rust/Cargo.
- Security/reliability: local controls and regression checks verified; production
  security review remains required.
- End-to-end: sensor-only and NLP-only local paths covered; genuine fusion and
  live infrastructure scenarios blocked.
- Documentation: current status and commands recorded in this handover.
- Performance: warmed CPU-only sensor inference measured approximately 3.569 ms;
  transformer/fusion/live-service/desktop benchmarks remain unavailable.
- Frontend dependency audit: zero vulnerabilities.

## Evidence

- Full Python suite after this phase: 429 passed, 4 skipped, 2 warnings;
  25 subtests passed. The focused assistant/API additions passed 19 tests and
  the EDP workbook audit tests passed 4 tests.
- Sensor baseline verifier: pass; frozen checkpoint and scaler load locally.
- Sensor embeddings: 17,731 x 64, complete index 0..17,730.
- Frontend Vite build: pass (1579 modules; 171.14 kB JavaScript, 54.27 kB gzip).
- Rust/Tauri: not verified because Rust/Cargo are unavailable on PATH.
- Official FD001 test status: `not_run`; validation metrics remain separate.
- Transformer NLP weights: unavailable; rule-based NLP remains available.
- MQTT/Influx: local code and replay tests pass; live infrastructure not tested.
- Dashboard refresh now exposes initial loading/error states, fetches registered
  machine health evidence, and keeps backend/API configuration in
  `frontend/.env.example` for browser development. MQTT/Influx replay now
  suppresses duplicate `(unit_id, cycle, event_timestamp)` identities before
  persistence. Machine PATCH requests reject unknown or identity-changing
  fields.

## Main runtime commands

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
cd frontend
npm run build
```

## Verification commands

```powershell
python -m pytest -q
python scripts/verify_lstm_baseline.py
python scripts/verify_preprocessing.py
python scripts/verify_sensor_embeddings.py
python scripts/verify_fusion.py
python scripts/verify_fusion_provenance.py
python scripts/verify_nlp.py
```

## API truthfulness and attachments

`GET /api/status` reports official-test evaluation as `not_run` unless the
explicit test metrics artifact is present. Validation metrics identify the
validation split. `GET /api/attachments/contract` is the shared upload contract;
TXT, CSV, JSON, PDF, JPEG, and PNG are validated for size, extension/MIME
agreement, and content before local storage.

## IoT limitation

Real sensor payloads should include timezone-aware `event_timestamp`. FD001
replay data has no physical timestamps and uses synthetic cycle-derived time only
for idempotent local storage. This is not a live timestamp claim.

## Known blockers

1. Authorized genuine technician-log source and provenance.
2. Approved sensor/log equipment crosswalk and cycle semantics.
3. Approved prediction cutoff and late-arrival policy.
4. Immutable local transformer model/tokenizer assets.
5. Genuine paired dataset and fusion authorization/checkpoint.
6. Rust/Cargo installation and clean-machine Tauri/NSIS verification.
7. Production authentication, authorization, rate limiting, and live service
   deployment decisions.

## Git state

This development phase does not commit or push changes. Existing local user
files are preserved; in particular, `frontend/tsconfig.tsbuildinfo` remains
untracked and excluded from project changes.
