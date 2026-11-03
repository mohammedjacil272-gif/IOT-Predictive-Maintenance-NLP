# Proposed Fusion Evaluation Protocol — Review Only

**Status: PENDING APPROVAL. Do not execute this protocol yet.** This document records decisions needed for a future controlled experiment. No training or performance evaluation is authorized by this proposal.

## Decisions requiring approval

| Decision | Proposed requirement | Status |
|---|---|---|
| Dataset | Approved genuine paired sensor telemetry and technician-written text with defensible point-in-time pairing | **PENDING APPROVAL** |
| Target | Explicitly defined RUL target and prediction cutoff/horizon, backed by approved target provenance | **PENDING APPROVAL** |
| Prediction horizon | Specify cycle/time horizon and operational use case | **PENDING APPROVAL** |
| Split | Equipment-level train/validation/test split with no equipment overlap; consider chronological ordering consistent with intended deployment | **PENDING APPROVAL** |
| Pairing | Join by approved `(unit_id, cycle)` semantics or another documented source-grounded key; never by array position | **PENDING APPROVAL** |
| Text availability | Include only fields available at the prediction cutoff; audit delayed and post-event work orders | **PENDING APPROVAL** |
| Missing text | Keep unmatched sensor windows and use the documented missing-text representation; report missingness by split | **PENDING APPROVAL** |
| Sensor-only comparator | Evaluate the frozen sensor-only model on exactly the same approved dataset split and target, without changing its preprocessing, scaler, features, checkpoint, or configuration | **PENDING APPROVAL** |
| Fusion model | Specify model version, feature construction, seed policy, and allowed training procedure after contract approval | **PENDING APPROVAL** |
| Metrics | Report MAE, RMSE, and MSE on the same held-out set; include sample counts and per-split coverage. Additional metrics require agreement | **PENDING APPROVAL** |
| Model selection | Predefine validation metric for checkpoint selection, tie-breaking, early stopping, and seed/repeat policy before seeing test results | **PENDING APPROVAL** |
| Leakage checks | Verify equipment split isolation, target exclusion from inputs, no future/terminal information, no post-cutoff text, and source-row provenance | **PENDING APPROVAL** |
| Reporting | Report the comparator and fusion results side by side, confidence/variability where applicable, missing-text coverage, limitations, and any deviations | **PENDING APPROVAL** |

## Required experiment sequence after approval

1. Freeze and version the approved paired dataset and data contract.
2. Run schema, provenance, pairing, point-in-time availability, and leakage checks before training.
3. Lock equipment-level splits and the prediction target/horizon.
4. Evaluate the frozen sensor-only comparator on the approved split and target without modifying its baseline artifacts.
5. Train/evaluate the fusion model only after explicit authorization and only on the approved training/validation partitions.
6. Select checkpoints using the approved validation rule; do not use the held-out test set for model selection.
7. Evaluate both models on the same held-out equipment/test partition and report identical metrics and sample coverage.
8. Publish no improvement claim unless the comparison is valid, reproducible, and approved by the project team.

## Explicit exclusions

- Do not use official FD001 test labels as training inputs or live inference features.
- Do not treat synthetic fixture values or OIL-039 synthetic probabilities as ground truth by default.
- Do not change the frozen sensor-only baseline to accommodate fusion.
- Do not run this protocol as part of documentation readiness work.
