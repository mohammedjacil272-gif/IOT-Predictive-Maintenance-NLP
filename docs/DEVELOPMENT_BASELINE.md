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

- `best_model.pt`: `21c059a7f87c151ef3f78db3a8af9e2fdd0441fb0214136fc1ae583ad5971d70` (`REGENERATED_CHECKPOINT_NOT_ORIGINAL`)
- `last_model.pt`: `76b00ce715d28aeadc8c812abc3f091a61ae0bd2506b529d0ce19d560814bc8f` (`REGENERATED_CHECKPOINT_NOT_ORIGINAL`)

The regenerated validation artifact reports MAE 20.749341, RMSE 31.943616, and MSE
983.757406 on the frozen validation split. Official-test status is `not_run`
unless an explicit test metrics artifact is supplied. The final official-test evaluation reports MAE 20.180132, RMSE 31.178255, and MSE 972.083587; these labels were not used for training or model selection.

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
