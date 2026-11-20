# Final integration audit

## Synchronization

`IOT_Integrated` was created as a separate worktree from fetched `origin/main` on local branch `integration/jacil-full-project`. The original `IOT_` directory was not reset, cleaned, overwritten, stashed, or merged. Sibling worktrees were inspected read-only. Current main already contains the strongest fusion/NLP contract implementation, so no sibling fusion source was blindly copied.

## Integrated work

The integration adds `app/main.py`, a static dashboard under `app/static/`, API tests, `requirements.txt`, `.env.example`, `start_backend.ps1`, and current architecture/status/contract/setup/limitation/handoff documents. Existing sensor, NLP, fusion, IoT, provenance, and verification implementations from `origin/main` remain in place. Local ignored FD001 `best_model.pt` and `last_model.pt` were copied byte-for-byte from the original local worktree only to make local inference reproducible in this worktree; source hashes are recorded in the prior audit and were not changed.

## Verification results

Before integration, the mainline suite had 364 passed, 3 skipped, 1 failure caused solely by missing ignored `best_model.pt`, and one pre-existing warning. After the checkpoint copy and API additions:

```text
368 passed, 3 skipped, 1 warning, 25 subtests passed in 39.90s
```

The warning is the existing `test_influx_ingestion.py::test_config` return-value warning.

Passed checks:

- `scripts/verify_preprocessing.py`: 17 features, 30-cycle windows, 17,731 training windows, 100 test windows, deterministic output, no warnings.
- `scripts/verify_lstm_baseline.py`: split/scaler/checkpoint/embedding/leakage checks passed.
- `scripts/verify_fusion.py` with `PYTHONPATH=.`: synthetic forward-only smoke test passed; it performs no training or artifact write.
- `scripts/verify_fusion_provenance.py`: synthetic provenance/alignment checks passed.
- Backend startup: Uvicorn startup completed; `GET /api/health` and `/` returned HTTP 200; `/api/status` reported database-disabled local mode.
- Official FD001 evaluation was run into new `artifacts/integration_eval_20261108/` only: 100 official test windows, MAE 17.7447274303, RMSE 23.9983521498, MSE 575.9209059038, R² 0.6664942704. Labels were used for final evaluation only and not training/model selection.

`scripts/verify_nlp.py` without a package argument reports blocked because its default flat-package layout does not match the current nested `artifacts/data/fd001_nlp_fixture` and `artifacts/embeddings/fd001_nlp_fixture` layout. The dedicated provenance and NLP/fusion tests pass; no fixture regeneration was performed.

## Final status

The project is locally runnable with a real backend/UI and sensor/NLP contract operation. Fusion is not presented as a production prediction: no approved real paired data, genuine training authorization, or trained validated fusion checkpoint exists. Transformer embeddings remain optional and unavailable without local assets. MQTT/Influx are not contacted by default.

## Start commands

```powershell
cd C:\Users\moham\Desktop\IOT_Integrated
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or:

```powershell
.\start_backend.ps1
```

## Remaining work

Approve real logs/mappings/targets/cutoffs, provide pinned transformer assets if needed, authorize and train fusion, define production auth/deployment, and run live IoT checks only when services are intentionally configured.
