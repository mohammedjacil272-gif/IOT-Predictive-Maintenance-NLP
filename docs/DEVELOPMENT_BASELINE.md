# Development baseline

## Repository state

- Repository: IOT-Predictive-Maintenance-NLP
- Audited commit at start of this phase: `aabe917`
- Branch: `main`, tracking `origin/main`
- Remote fetch found no incoming commits.
- Pre-existing untracked `frontend/tsconfig.tsbuildinfo` is preserved and is
  excluded from project changes.

## Runtime

The current environment reports Python 3.11.9, Node 22.20.0, npm 10.9.3,
PyTorch 2.15.0+cpu, and CUDA unavailable. Rust/Cargo are not installed on
this PATH, so Tauri checks cannot run in this environment.

## Baseline artifacts

The frozen FD001 baseline is loaded read-only. Known SHA-256 values:

- `best_model.pt`: `388d6e45f6d9692aee1d8cc9d8de021c4fab4eb0ce8287445eb24973cc2db013`
- `last_model.pt`: `01bd4e6b114af77ad2eb3e4dc6088389b8522b6b0f1a1b5e7ba704857a3d5615`

The validation artifact reports MAE 20.884394, RMSE 31.364907, and MSE
983.757406 on the frozen validation split. Official-test status is `not_run`
unless an explicit test metrics artifact is supplied.

## Revalidation commands

```powershell
python -m pytest -q
python scripts/verify_lstm_baseline.py
python scripts/verify_preprocessing.py
python scripts/verify_sensor_embeddings.py
cd frontend; npm run build
```

The baseline architecture, weights, preprocessing arrays, and embeddings must
not be overwritten by development work.
