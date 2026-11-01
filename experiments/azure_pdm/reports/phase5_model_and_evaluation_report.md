# Phase 5 — Azure PdM Model and Evaluation Report

Status: Completed authorized training/evaluation on the simulated benchmark

## 1. Data-integrity verdict

**PASS.** The canonical Phase 4 v3 dataset passed an independent pre-training gate before any model fitting.

Verified independently:

- Canonical dataset hash matched the Phase 4 manifest: `431af6f141d559f30b64c9b0ec3295b2429c11b7a6188c6aca26398f6c60caf9`.
- 575,300 rows total.
- 288,000 unique `(machineID, anchor_timestamp)` pairs.
- 288,000 3-hour rows plus 287,300 eligible 24-hour rows = 575,300.
- 575,300 unique `(machineID, anchor_timestamp, lookback_hours)` keys and sample IDs.
- Target classes recomputed independently from raw failures with zero mismatches.
- Target-event timestamps and provenance had zero mismatches.
- Split assignment had zero mismatches.
- Raw failure duplicate keys: 0.
- All six classes are present in train, validation, and test.
- Raw source hashes still match the acquisition manifest.

No failure-derived values, event features, maintenance features, timestamps, sample IDs, or split identifiers were used as model inputs.

## 2. Anchor, lookback, and split audit

Telemetry coverage is `2015-01-01T06:00:00` through `2016-01-01T06:00:00`, with 876,100 hourly rows across 100 machines. Each machine has 8,761 expected and observed hourly timestamps; no internal hourly gaps were found.

Lookback configurations:

| Configuration | Rows | Input shape |
|---|---:|---|
| 3-hour | 288,000 | 3 hourly observations × 4 telemetry fields |
| 24-hour | 287,300 | 24 hourly observations × 4 telemetry fields |

Windows use `[t-lookback,t)` and contain only actual telemetry observations. No interpolation, forward-fill, backward-fill, or invented anchors was used.

Split anchors use the v1.1 two-sided 24-hour purge:

- Train: `anchor < 2015-08-07 06:00:00`
- Validation: `2015-08-09 06:00:00 <= anchor < 2015-10-19 06:00:00`
- Test: `anchor >= 2015-10-21 06:00:00`

Rows by configuration and split:

| Lookback | Train | Validation | Test |
|---|---:|---:|---:|
| 3h | 174,300 | 56,800 | 56,900 |
| 24h | 173,600 | 56,800 | 56,900 |

Independent checks found no target interval crossing into a later split, no duplicate configuration keys, and no machine appearing under conflicting split assignments.

## 3. Target and class distribution

Across both lookbacks:

| Class | Rows |
|---|---:|
| `none` | 564,111 |
| `comp1` | 2,638 |
| `comp2` | 3,596 |
| `comp3` | 1,837 |
| `comp4` | 2,450 |
| `multi_failure` | 668 |

The target interval is `(t,t+24h]`. `multi_failure` represents at least two distinct components in that interval. Same-component repeats do not automatically become `multi_failure`. Duplicate same-machine/component/timestamp failures were absent and are fail-closed by the source audit.

## 4. Model architecture and inputs

Implemented under `experiments/azure_pdm/`:

- `model.py`: compact temporal CNN classifier.
- `train.py`: reproducible preparation, baselines, training, checkpointing, and evaluation.

Architecture:

1. Two-layer 1D temporal CNN over `[time, volt, rotate, pressure, vibration]`.
2. Adaptive temporal pooling.
3. Machine metadata branch containing standardized age and train-fitted one-hot model category.
4. Fusion layer.
5. Six-class classification head.

Errors, maintenance, failure records, target provenance, synthetic text, and BERT embeddings were excluded. The existing FD001 RUL regression fusion module was not reused or modified.

Preprocessing was fit on training rows only:

- telemetry mean and standard deviation;
- machine-age mean and standard deviation;
- machine-model vocabulary.

Configuration:

- Seed: 42.
- Device: CPU.
- Optimizer: Adam, learning rate `1e-3`.
- Batch size: 2,048.
- Requested epochs: 3.
- Class-weighted cross-entropy using training-only class counts.
- Best checkpoint selected by validation macro F1.

## 5. Baseline results

Results below are on the untouched test partition.

| Lookback | Baseline | Accuracy | Macro F1 | Balanced accuracy | Weighted F1 | Macro OVR AP |
|---|---|---:|---:|---:|---:|---:|
| 3h | Majority `none` | 0.9820 | 0.1652 | 0.1667 | 0.9731 | 0.1667 |
| 3h | Balanced logistic regression | 0.7008 | 0.2129 | 0.7093 | 0.8100 | 0.2482 |
| 24h | Majority `none` | 0.9820 | 0.1652 | 0.1667 | 0.9731 | 0.1667 |
| 24h | Balanced logistic regression | 0.9112 | 0.3792 | 0.8986 | 0.9413 | 0.3656 |

The logistic baseline emitted a bounded `lbfgs` convergence warning at 100 iterations. Its recorded results are retained and are not presented as fully converged.

## 6. Neural model results

| Lookback | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Balanced accuracy | OVR AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3h | 0.6157 | 0.1958 | 0.6904 | 0.1824 | 0.7472 | 0.6904 | 0.2369 |
| 24h | 0.8798 | 0.2576 | 0.8569 | 0.3167 | 0.9230 | 0.8569 | 0.3238 |

Per-class neural metrics:

### 3-hour model

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `none` | 0.9983 | 0.6129 | 0.7595 | 55,876 |
| `comp1` | 0.0237 | 0.7543 | 0.0460 | 232 |
| `comp2` | 0.0403 | 0.7820 | 0.0767 | 344 |
| `comp3` | 0.0576 | 0.8929 | 0.1081 | 168 |
| `comp4` | 0.0457 | 0.8365 | 0.0866 | 208 |
| `multi_failure` | 0.0089 | 0.2639 | 0.0172 | 72 |

### 24-hour model

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `none` | 0.9999 | 0.8799 | 0.9361 | 55,876 |
| `comp1` | 0.1167 | 0.8750 | 0.2059 | 232 |
| `comp2` | 0.1434 | 0.8721 | 0.2463 | 344 |
| `comp3` | 0.0957 | 0.9345 | 0.1737 | 168 |
| `comp4` | 0.1213 | 0.8990 | 0.2138 | 208 |
| `multi_failure` | 0.0683 | 0.6806 | 0.1242 | 72 |

Per-component recall for the neural models:

| Lookback | comp1 | comp2 | comp3 | comp4 | multi_failure precision | multi_failure recall |
|---|---:|---:|---:|---:|---:|---:|
| 3h | 0.7543 | 0.7820 | 0.8929 | 0.8365 | 0.0089 | 0.2639 |
| 24h | 0.8750 | 0.8721 | 0.9345 | 0.8990 | 0.0683 | 0.6806 |

The 24-hour model is stronger than the 3-hour model on macro F1, balanced accuracy, weighted F1, and average precision. Both neural models underperform the balanced logistic baseline at this short three-epoch run, and both show severe rare-class precision limitations. The class-weighting strategy improves minority recall but produces many false positives.

## 7. Confusion matrices

Rows are actual classes and columns are predicted classes in this order:

`none, comp1, comp2, comp3, comp4, multi_failure`

### 3-hour neural model

```text
[[34246, 7170, 6381, 2433, 3613, 2033],
 [30,    175,    2,    7,    4,   14],
 [21,      4,  269,   11,    2,   37],
 [1,       1,    2,  150,    2,   12],
 [3,       4,    1,    1,  174,   25],
 [2,      15,   17,    4,   15,   19]]
```

### 24-hour neural model

```text
[[49165, 1529, 1792, 1483, 1333, 574],
 [5,     203,    0,    0,    6,  18],
 [0,       0,  300,    0,    0,  44],
 [0,       0,    0,  157,    0,  11],
 [0,       0,    0,    0,  187,  21],
 [0,       8,    0,    0,   15,  49]]
```

The dominant error pattern is false-positive assignment of `none` rows to rare classes, especially for the 3-hour model. `multi_failure` support is only 72 test rows, so its metrics are high variance.

## 8. Training history and checkpoint selection

Both configurations completed the bounded three-epoch run. The best checkpoint for both was selected at epoch 3 using validation macro F1.

The first attempt in `phase5_outputs` was stopped before model fitting because the installed scikit-learn version rejected the deprecated `multi_class` argument. That partial output was preserved. A corrected run was completed in `phase5_outputs_02`; `phase5_outputs_03` is the final canonical run because it additionally records dependency and split provenance in its manifest.

## 9. Outputs and provenance

Canonical output root:

`${AZURE_PDM_ROOT}\phase5_outputs_03`

Artifacts:

- 3-hour checkpoint: `lookback_3h\best_model.pt`
  - SHA-256: `7374989a5c28f8edaaecfdaa1fb309de8195538ddb061efbf9ddd4ac3977c554`
- 24-hour checkpoint: `lookback_24h\best_model.pt`
  - SHA-256: `7b326c118e997ed82bbc53459bc98a7da764e6e65b55d598238ee8c1f00ce59e`
- Per-lookback `evaluation.json`, `training_history.json`, `preprocessing.json`, and `preprocessed_arrays.npz`.
- `phase5_manifest.json` records contract version `azure_pdm_experiment_contract_v1.1`, raw source hashes, acquisition-manifest hash, derived dataset hash, split configuration, feature order, dependency versions, lookback configuration, and checkpoint results.

## 10. Tests

Final runnable regression selection:

```text
152 passed, 1 warning
```

The warning is the pre-existing pytest return-value warning in `tests/test_influx_ingestion.py::test_config`.

The known `tests/test_training_contract.py` collection issue remains blocked because `training.train_multimodal` is absent. No replacement module was invented and no test was altered to hide the failure.

## 11. Limitations and remaining blockers

- This is simulated benchmark data, not factory data.
- Telemetry timestamps use the approved benchmark-only observation-time assumption.
- Errors and maintenance remain unavailable as features because event availability is unknown.
- No technician narratives or BERT embeddings were used.
- The neural run was intentionally bounded to three epochs on CPU; results are not a hyperparameter-optimized benchmark.
- Logistic regression did not converge within its bounded 100 iterations.
- Rare-class precision, especially `multi_failure`, is weak.
- The model inputs are causal telemetry windows plus machine metadata; no event modality was evaluated.
- No performance claim may be generalized to real-world predictive maintenance.

## 12. Final safety statement

Raw CSVs and acquisition manifest remain hash-identical. FD001 artifacts and reports were not modified. Existing IoT production modules and untracked files were preserved. No prior Phase 4 output was overwritten; `phase5_outputs` is retained as the partial failed attempt, `phase5_outputs_02` is retained as a corrected run, and `phase5_outputs_03` is the canonical completed run.

Training and test evaluation did occur under the approved Azure experiment scope. No inference service was run, no FD001 artifact was reused, and no real-world performance claim was made.
