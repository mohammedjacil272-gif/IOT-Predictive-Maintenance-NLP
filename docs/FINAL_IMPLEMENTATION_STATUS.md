# MaintainIQ implementation status

Refresh: 2026-11-09. This is the final status of the canonical D: worktree.
Statuses use the requested vocabulary and distinguish executable interfaces
from genuine-data or external-service validation.

| Phase | Implementation | Tests | Status | Blockers |
|---|---|---|---|---|
| 1. Repository/data audit | Audit tooling, schema tests, repository inspection, protected-artifact checks | Full suite; EDP schema tests; direct CLI gate | COMPLETE | EDP raw files unavailable |
| 2. Dataset research/acquisition | Candidate review, selection decision, lawful acquisition checklist | Dataset registry tests; no raw acquisition claim | PARTIALLY COMPLETE | Approved paired dataset/license/access |
| 3. Data contract/ingestion | CSV/Excel contract pieces, MQTT/Influx validation, provenance-aware schemas | Ingestion, sensor, registry tests | COMPLETE | Live services not connected |
| 4. Alignment/leakage | Fusion alignment/provenance policies, split isolation, missing-modality semantics | Alignment, provenance, leakage-related tests | COMPLETE | Genuine pair validation awaits data |
| 5. Sensor/NLP baselines | Frozen FD001 sensor baseline; rule-based NLP and fixture embeddings | Baseline/NLP/artifact verifiers | PARTIALLY COMPLETE | Transformer revisions and genuine text unavailable |
| 6. Multimodal fusion | Fusion adapters, masks, smoke path, provenance gates | Fusion smoke/provenance/compatibility tests | PARTIALLY COMPLETE | No genuine aligned sensor-text corpus or trained fusion checkpoint |
| 7. IoT/realtime | Validated local replay, MQTT/Influx adapter, inference path, duplicate suppression | IoT and replay tests | PARTIALLY COMPLETE | No broker, InfluxDB, or industrial hardware |
| 8. Backend/frontend/Tauri | FastAPI routes, constrained machine PATCH, React dashboard health/loading states, API integration, frontend build | API tests, TypeScript, Vite build | PARTIALLY COMPLETE | Rust/Cargo/NSIS unavailable for desktop build |
| 9. AI assistant | Grounding, retrieval boundaries, allowlisted read-only tools, confirmation writes, attachments | Assistant/API/security tests | PARTIALLY COMPLETE | Ollama unavailable; live LLM inference not demonstrated |
| 10. Security/performance/handover | Security review, local benchmark, full validation and documentation | 429 passed, 4 skipped, 2 warnings; npm audit clean; diff check clean | PARTIALLY COMPLETE | Production auth/rate limits/live benchmarks/external scans remain |

## Git and protected state

- Branch: `main`; HEAD: `aabe917b7a123d1f8cc90618afe18b49c3011070`.
- Existing modified and untracked work was preserved. The PR13-compatible
  verifier patch and the EDP auditor path bootstrap were applied without
  replacing unrelated implementation.
- The PR13 compatibility change is present in
  `scripts/verify_nlp_fd001_compatibility.py` and matches
  `origin/pr13-resolution` for that file.
- Frozen `artifacts/preprocessing/fd001/` artifacts were not modified.
- No commit, push, merge, rebase, branch switch, reset, clean, or deletion was
  performed.

## Data and models

The repository contains NASA C-MAPSS FD001 data and explicitly namespaced
synthetic NLP/fusion fixtures. No EDP raw workbook or authorized genuine paired
sensor-technician dataset is present. The sensor baseline and its 64-D
embeddings are verified; official FD001 test evaluation remains `not_run`.
Fusion and NLP fixture checks are interface/smoke validation, not genuine
multimodal performance evidence.

## Remaining handover actions

Acquire and approve a paired dataset, record hashes/licenses, run the EDP audit,
approve temporal cutoffs, install approved model assets if desired, connect a
test MQTT/Influx environment, install Rust/Cargo/NSIS, and perform deployment
authentication/rate-limit/dependency reviews before production claims.
