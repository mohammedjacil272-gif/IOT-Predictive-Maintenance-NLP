# MaintainIQ multimodal handover

Date: 2026-11-09
Status: **SENSOR-ONLY DEMONSTRATION READY; GENUINE MULTIMODAL DATA BLOCKED**

## Current evidence

- Repository remains on `main` at `aabe917` with preserved uncommitted work.
- Protected FD001 hashes match; no protected baseline files changed.
- Latest full suite after dashboard, API-contract, and IoT duplicate-protection
  changes: 429 passed, 4 skipped, 2 warnings, 25 subtests.
- EDP source metadata was independently inspected online, but raw workbooks
  were not obtained because download/API access was unavailable from this
  environment. No actual EDP schema, text, or row-count claim is made.
- EDP read-only audit tooling and tests were added; no model training ran.
- Genuine multimodal fusion remains blocked. Synthetic fusion tests are not
  performance evidence.
- Frontend TypeScript and production build passed in the final refresh; the
  production dependency audit reports zero vulnerabilities. Tauri remains
  unverified without Rust/Cargo.
- Assistant grounding and confirmation safety are implemented; live Ollama and
  genuine-data end-to-end grounding were not demonstrated.

## Blockers and next actions

1. Obtain the lawful EDP release or an approved alternative; record raw hashes.
2. Install an Excel engine and run `scripts/audit_edp_dataset.py`.
3. Review remarks, IDs, timestamps, failure semantics, and attribution.
4. Approve the alignment/cutoff/split manifest before training.
5. Install approved local transformer/Ollama assets if desired.
6. Train and evaluate independent sensor/text/fusion models only after gates.
7. Install Rust/Cargo/NSIS for desktop verification.

## Final validation evidence

- `python -m pytest -q`: 427 passed, 4 skipped, 2 warnings, 25 subtests.
- FD001 preprocessing, LSTM, sensor embedding, fusion, NLP compatibility, and
  artifact handover verifiers passed. NLP fixture validation remains explicitly
  non-genuine because provenance, immutable model revisions, timestamps, and
  signed authenticity evidence are absent.
- `npx tsc -b` and `npm run build` passed. Cargo and Ollama were unavailable.
- The EDP CLI correctly rejected the missing `Data/external_datasets/edp/raw`
  prerequisite; no workbook audit or acquisition claim was made.
- The React dashboard now shows backend loading/error states and registered
  machine health evidence. Browser development API configuration is documented
  in `frontend/.env.example`; Tauri still supplies its managed sidecar URL and
  token when available.
- MQTT/Influx ingestion now reports and skips duplicate message identities;
  this was validated with a deterministic replay test.

## Commands

```powershell
python -m pytest -q
python scripts/audit_edp_dataset.py Data/external_datasets/edp/raw --report reports/dataset_validation/edp_audit.json
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
