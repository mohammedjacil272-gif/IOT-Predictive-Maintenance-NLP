# FD001 artifact handover

Status: **regenerated checkpoint locally verified; fresh-clone distribution blocked by artifact-storage policy**.

The local handover contains a clearly labeled `REGENERATED_CHECKPOINT_NOT_ORIGINAL` replacement, the two checkpoints, preprocessing arrays, and 64-D sensor embeddings required for sensor-only inference. The read-only
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

The replacement was trained in `artifacts/recovery/fd001_lstm_recovered_20261110` with seed 42, 80/20 engine isolation, 20 maximum epochs, batch size 128, and validation-RMSE checkpoint selection. The previous application metadata was backed up under that recovery directory before promotion. Official FD001 labels were used only for the explicit final evaluation, never for training or checkpoint selection. A missing or mismatched artifact fails closed with its expected and actual path/hash.
