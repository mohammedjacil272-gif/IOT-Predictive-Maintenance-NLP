# FD001 artifact handover

Status: **locally reproducible and verified; fresh-clone distribution blocked by
artifact-storage policy**.

The frozen local handover contains the two checkpoints, preprocessing arrays,
and 64-D sensor embeddings required for sensor-only inference. The read-only
verification command checks the recorded SHA-256 values, model configuration,
checkpoint state dict compatibility, preprocessing metadata, and one finite
validation-window prediction:

```powershell
python scripts/verify_fd001_artifact_handover.py
```

The protected artifact hashes are recorded in `docs/DEVELOPMENT_BASELINE.md`.
The files are intentionally excluded from normal Git tracking because they are
large model/numeric artifacts. A fresh clone therefore requires a separately
versioned artifact transfer or approved Git LFS/external registry decision.
The transfer must preserve the following relative paths under `artifacts/`:

- `models/fd001_lstm_baseline/best_model.pt`
- `models/fd001_lstm_baseline/last_model.pt`
- `models/fd001_lstm_baseline/model_config.json`
- `data/fd001_lstm_baseline/baseline_preprocessing_config.json`
- `data/fd001_lstm_baseline/X_train.npy`
- `data/fd001_lstm_baseline/X_val.npy`
- `data/fd001_lstm_baseline/X_test.npy`
- `embeddings/fd001_lstm_baseline/sensor_embeddings.npy`

No checkpoint is recreated, overwritten, retrained, or added to Git by this
handover. A missing or mismatched artifact fails closed with its expected and
actual path/hash.
