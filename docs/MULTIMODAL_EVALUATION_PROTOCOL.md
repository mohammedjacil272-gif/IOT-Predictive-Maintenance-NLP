# Proposed Multimodal Evaluation Protocol — NOT AUTHORIZED TO RUN

**Status:** Planning only. Every domain choice below is **PENDING APPROVAL**. This is not authorization to train or evaluate the fusion model.

## Approval prerequisites

Before any genuine fusion run, obtain explicit approval for:

1. Permissioned genuine paired sensor/technician dataset and immutable dataset version.
2. Stable equipment namespace/crosswalk and shared cycle/window semantics.
3. Timestamp/timezone meanings, authoritative log availability field, prediction cutoff and exclusion rule for any log unavailable at cutoff.
4. RUL target definition/provenance, prediction cycle, cutoff and operational horizon.
5. Equipment-level train/validation/test assignment with disjoint engine sets and versioned split manifest.
6. Provenance schema and leakage checks.
7. Missing-text policy, multiple-log aggregation rule and whether `has_log` is a model input.
8. Reproducible dependencies, approved text-model revision, checksums, and baseline checkpoint availability.
9. Approved comparison protocol and immutable held-out test population.

## Proposed split and integrity controls

- Split by equipment, not random windows; no equipment ID may occur across train, validation, and test.
- Persist equipment IDs and split-manifest hash before experiments.
- Fit learned transforms only on training equipment. Do not modify frozen baseline preprocessing, scaler, 17-feature order, 30-cycle window, checkpoint, or metrics.
- Join by approved `(unit_id, cycle)` semantics after verifying the real equipment mapping.
- Include technician text only when authoritative metadata proves it was available at/before prediction cutoff. Unknown availability is a no-go unless an approved policy explicitly excludes the record.
- Exclude RUL targets, terminal/future cycles, future failure outcomes and post-cutoff fields from inference inputs.
- Preserve per-sample source lineage and report missing-text coverage by split.

## Model selection and final comparison — PENDING APPROVAL

1. Run schema, source provenance, key uniqueness, point-in-time availability, split and leakage checks before any training.
2. Fit on training equipment only.
3. Use validation equipment only for checkpoint selection and approved hyperparameter choices.
4. Reserve a held-out test equipment set for a final comparison after all choices are frozen.
5. Compare fusion and frozen sensor-only baseline on the same eligible held-out equipment, target, cutoff and metric definition. If eligible rows differ, disclose the difference rather than silently comparing different populations.
6. Candidate metrics are MAE, RMSE and MSE, subject to approval. Report sample/engine counts, missing-text coverage, target distribution, split identity, seeds, runtime, checkpoint/model hashes, provenance and leakage-check results.
7. Predefine repeated seeds/uncertainty handling, stopping/model-selection policy and how failed runs are reported. Do not use test results for tuning.
8. Report limitations and distinguish synthetic interface verification from genuine performance evidence.

## No-go conditions

Do not evaluate until source permissions, pairing, availability/cutoff, target/horizon, held-out equipment, baseline checkpoint identity, split integrity, provenance and model/environment reproducibility are resolved. Never use official FD001 test labels for training/model selection/live features. Do not fabricate a mapping between FD001 and OIL-039.
