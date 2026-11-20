# FD001 sensor-only model card

Status: implemented and locally verified; frozen baseline.

- Task: RUL regression on NASA C-MAPSS FD001.
- Input: 30 cycles x 17 selected features.
- Architecture: two-layer LSTM, hidden size 64, dropout 0.2, regression head.
- Split: equipment-level 80/20 train/validation, seed 42.
- Checkpoint selection: lowest validation RMSE.
- Validation metrics: MAE 20.884394, RMSE 31.364907, MSE 983.757406.
- Official FD001 test evaluation: `not_run` in the current API state.

The checkpoint and preprocessing artifacts are local handover artifacts and are
not replaced, retrained, or represented as official-test results.
