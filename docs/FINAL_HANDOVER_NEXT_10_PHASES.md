# MaintainIQ — next ten phases handover

Date: 2026-11-09
Status: **READY FOR SENSOR-ONLY DEMONSTRATION — MULTIMODAL DATA BLOCKED**

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 1. Dataset research | Partially complete | `DATASET_RESEARCH_AND_SELECTION.md`; no raw download, no authorized paired corpus accepted |
| 2. Ingestion/validation | Implemented and verified | Azure adapter, `preprocessing/dataset_registry.py`, focused tests |
| 3. FD001 reproducibility | Implemented and verified | protected hashes and handover verifier |
| 4. NLP pipeline | Implemented baseline; transformer unavailable | rule NLP, artifact validator, duplicate-key tests |
| 5. Alignment/fusion | Interfaces verified; genuine training blocked | `MULTIMODAL_EXPERIMENT_REPORT.md`, synthetic-only smoke tests |
| 6. IoT | Implemented contracts; local/mock verification only | timezone-aware event timestamp policy and dry-run verifier |
| 7. Application integration | Backend/frontend verified; Tauri unverified | frontend build passes; Rust/Cargo unavailable |
| 8. Maintenance assistant | Local foundation and safety boundary implemented; Ollama optional | gateway, retrieval, health, confirmation, grounding tests |
| 9. Security/performance/E2E | Partially complete | path/resource checks, npm audit, CPU smoke benchmark, failure-state tests |
| 10. Documentation | Complete for actual state | this handover and linked reports |

## Data and ML decisions

No external data was downloaded. The Swedish paper-machine dataset is the best
public research lead for sensor-plus-technical-language work, but its license,
download terms, source semantics, language model, and prediction cutoff must be
approved before intake. Azure is a structured benchmark/sample and is not
treated as free-text technician evidence. Synthetic datasets remain fixtures.

The six protected FD001 hashes matched. The sensor embedding is 17,731 x 64.
The historical FD001 validation RMSE is approximately 31.364907; official test
evaluation remains `not_run`. No genuine NLP embedding, fusion checkpoint,
multimodal metric, or comparison is claimed.

## Application and assistant

FastAPI status/readiness, attachment validation, local sensor inference, rule NLP,
React dashboard, and confirmation-gated assistant workflows are available.
Ollama is only used when an already-installed local model is configured. The
assistant does not execute destructive actions and treats retrieved text as
untrusted evidence. MQTT/Influx integration is contract-tested/dry-run only.

## Verification evidence

The latest full suite is **427 passed, 4 skipped, 2 warnings, 25 subtests
passed**. The frontend production build passed and `npm audit --omit=dev
--audit-level=high` reported 0 vulnerabilities. Rust/Cargo/Tauri checks remain
unavailable because the tools are not on PATH.

## Remaining blockers

1. Authorized genuine paired sensor/log data, mapping, cutoff policy, and
   provenance approval are required for genuine fusion training.
2. A verified local transformer model/tokenizer and approved corpus are needed
   for transformer embeddings and NLP metrics.
3. Ollama model installation is optional and machine-local.
4. Rust/Cargo/NSIS and an isolated Windows clean-machine test are required for
   a desktop installer claim.
5. Authorized local MQTT/Influx/Mongo services are required for live integration.

## Exact commands

```powershell
cd D:\Wahid-ML-Setup\IOT-Predictive-Maintenance-NLP
python -m pytest -q
python scripts/verify_fd001_artifact_handover.py
python scripts/verify_lstm_baseline.py
python scripts/verify_preprocessing.py
python scripts/verify_sensor_embeddings.py
python scripts/verify_fusion.py
python scripts/verify_fusion_provenance.py
python scripts/verify_nlp.py
cd frontend
npm run build
npm audit --omit=dev --audit-level=high
```

Do not commit or push this handover without the repository owner's explicit
authorization.
