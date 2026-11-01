# Phase 6 — Azure PdM optimization, generalization, and model selection

## Executive result

The mandatory data-integrity and leakage checks passed for the canonical Phase 4 v3 dataset. Bounded optimization and evaluation ran on CPU for both approved lookback configurations. The best neural candidate by validation macro F1 was a compact 24-hour aggregate MLP with capped class weights. It was not the best overall benchmark model: the preserved balanced logistic regression baseline remained stronger on the untouched chronological test set (macro F1 0.3792 versus 0.3286 for the selected neural candidate). The logistic baseline therefore remains the preferred simple benchmark reference; the MLP is retained as the best neural candidate for future review.

This is a simulated benchmark. These results are not real-world factory-performance claims.

## Scope, provenance, and preservation

- Repository branch: `feature/jacil-fusion-integration`
- HEAD at verification: `1b6770f106f34dae5b52e225fbef8813d02f33fc`
- Governing contract: `azure_pdm_experiment_contract_v1.1`
- Canonical derived dataset: `${AZURE_PDM_ROOT}\derived_phase4_v3\phase4_temporal_dataset.csv`
- Derived dataset SHA-256: `431af6f141d559f30b64c9b0ec3295b2429c11b7a6188c6aca26398f6c60caf9`
- Source hashes were rechecked against the acquisition manifest and matched.
- New external output root: `${AZURE_PDM_ROOT}\phase6_outputs`
- Reproducibility manifest: `${AZURE_PDM_ROOT}\phase6_outputs\phase6_manifest.json`

The prior `phase5_outputs_03` directory and its checkpoints were preserved. The canonical checkpoint hashes remained:

- 3h: `7374989a5c28f8edaaecfdaa1fb309de8195538ddb061efbf9ddd4ac3977c554`
- 24h: `7b326c118e997ed82bbc53459bc98a7da764e6e65b55d598238ee8c1f00ce59e`

No raw CSV, acquisition manifest, FD001 artifact, existing production module, or prior report was modified.

## Data-integrity gate

The canonical dataset contains 575,300 rows: 288,000 3-hour-lookback rows and 287,300 24-hour-lookback rows. The two lookbacks are separate rows/configurations, not duplicated model inputs within one configuration. There are 288,000 unique `(machineID, anchor_timestamp)` pairs and 575,300 unique `(machineID, anchor_timestamp, lookback_hours)` keys. The 700-row difference is the documented loss of 24-hour windows at the beginning of machine timelines.

Independent checks confirmed:

- Target interval is `(anchor, anchor + 24 hours]`.
- Recomputed target classes matched all stored targets.
- All six classes are represented in train, validation, and test.
- Target provenance timestamps matched source failure records.
- Duplicate failure keys were zero in the audited source.
- Split assignments matched the contract cutpoints and embargo policy.
- No duplicate sample IDs or machine/anchor/lookback combinations were found.
- Feature inputs contain telemetry and machine metadata only; failure provenance is not used as a feature.
- Missing telemetry was not interpolated or forward-filled.
- Preprocessing was fitted from training rows only.

The natural class total across the dataset was: `none` 564,111; `comp1` 2,638; `comp2` 3,596; `comp3` 1,837; `comp4` 2,450; `multi_failure` 668. The chronological test partition contained 56,900 rows for each configuration; its class support was `none` 55,876, `comp1` 232, `comp2` 344, `comp3` 168, `comp4` 208, and `multi_failure` 72.

## Experiment design

The bounded sweep used the same data interface, feature definitions, target mapping, and chronological split for both lookbacks. It compared the existing temporal CNN, a compact LSTM, a compact GRU, and a compact aggregate MLP. Each neural candidate used seed 42, two epochs, batch size 4,096, Adam, and validation macro F1 checkpoint selection. The sweep compared unrestricted inverse-frequency weights (`weighted`) with weights capped at 20 (`capped20`). No validation or test oversampling was used. Focal loss and sampling were not run in this bounded package.

The MLP aggregates each telemetry channel over the elapsed-time window and fuses the resulting statistics with machine model/age metadata. It is intentionally simpler than adding unnecessary sequence complexity. The new implementations are isolated in `experiments/azure_pdm/architectures.py` and `experiments/azure_pdm/optimize.py`; the FD001 RUL model was not modified or used as a drop-in classifier.

## Architecture sweep

Values are validation macro F1 / untouched test macro F1 / test balanced accuracy.

| Lookback | Architecture | Weighting | Validation F1 | Test F1 | Test balanced accuracy |
|---|---|---:|---:|---:|---:|
| 3h | CNN | weighted | 0.0765 | 0.0715 | 0.5913 |
| 3h | CNN | capped20 | 0.1910 | 0.1860 | 0.6363 |
| 3h | LSTM | weighted | 0.0473 | 0.0393 | 0.2944 |
| 3h | LSTM | capped20 | 0.2072 | 0.2031 | 0.4351 |
| 3h | GRU | weighted | 0.0960 | 0.0917 | 0.5279 |
| 3h | GRU | capped20 | 0.1947 | 0.1940 | 0.5344 |
| 3h | MLP | weighted | 0.0789 | 0.0847 | 0.6192 |
| 3h | MLP | capped20 | 0.2375 | 0.2308 | 0.6023 |
| 24h | CNN | weighted | 0.2218 | 0.2196 | 0.7737 |
| 24h | CNN | capped20 | 0.2956 | 0.2933 | 0.7724 |
| 24h | LSTM | weighted | 0.0516 | 0.0466 | 0.3881 |
| 24h | LSTM | capped20 | 0.1976 | 0.1985 | 0.7181 |
| 24h | GRU | weighted | 0.1331 | 0.1320 | 0.6117 |
| 24h | GRU | capped20 | 0.2748 | 0.2760 | 0.6592 |
| 24h | MLP | weighted | 0.1035 | 0.1223 | 0.6637 |
| 24h | MLP | capped20 | **0.3434** | **0.3286** | 0.6556 |

Capped weighting improved validation macro F1 for every neural architecture in this sweep, but did not solve the rare-class precision problem. The neural models generally traded precision for minority recall.

## Baseline comparison and selected checkpoint

The preserved balanced logistic baseline scored:

| Lookback | Validation macro F1 | Test macro F1 | Test balanced accuracy | Test multi-failure P/R |
|---|---:|---:|---:|---:|
| 3h | 0.2223 | 0.2129 | 0.7093 | 0.0136 / 0.4444 |
| 24h | 0.3836 | **0.3792** | 0.8986 | 0.1255 / 0.8056 |

The selected neural checkpoint is:

`${AZURE_PDM_ROOT}\phase6_outputs\lookback_24h\mlp_capped20\best_model.pt`

SHA-256: `4a79d3937f73138769a188c9f9e86f9b3489b50496d654f8e32c2f9ff3ef5efa`.

Its validation macro F1 was 0.3434. Standard test metrics were accuracy 0.9244, macro precision 0.2683, macro recall/balanced accuracy 0.6556, macro F1 0.3286, weighted F1 0.9481, and macro one-vs-rest average precision 0.3206. Per-class test F1 was: none 0.9611, comp1 0.2785, comp2 0.2633, comp3 0.1828, comp4 0.2862, and multi_failure 0.0000. Per-component recall was comp1 0.6207, comp2 0.9767, comp3 0.5238, and comp4 0.8846. Multi-failure precision and recall were both 0.0.

The selected standard test confusion matrix, with rows actual and columns predicted in the order `none, comp1, comp2, comp3, comp4, multi_failure`, was:

```text
[[51846, 649, 1812, 690, 879, 0],
 [82, 144, 4, 2, 0, 0],
 [1, 0, 336, 7, 0, 0],
 [79, 0, 1, 88, 0, 0],
 [8, 0, 15, 1, 184, 0],
 [1, 9, 40, 7, 15, 0]]
```

Because the 24h logistic baseline is higher on test macro F1, balanced accuracy, and multi-failure performance, the simpler baseline remains the preferred benchmark model. This comparison is reported after the validation-based neural selection and does not change the untouched test partition.

## Thresholds and calibration

Thresholds were tuned on validation only. The selected thresholds were `[1.0, 1.0, 1.0, 1.0, 1.0, 0.1]` for the class order above. Validation macro F1 changed from 0.3434 to 0.3436; tuned test macro F1 was 0.3289 and balanced accuracy remained 0.6556. Multi-failure precision and recall remained zero. Standard test expected calibration error was 0.6232, indicating poor calibration. No test information was used to select thresholds.

## Machine-held-out generalization

A separate complete-machine experiment used machines 1–60 for training, 61–80 for validation, and 81–100 for held-out test, with no machine IDs shared across partitions. It used the selected 24h MLP/capped20 configuration and refit preprocessing on the machine-training subset. Validation macro F1 was 0.3005; held-out-machine test macro F1 was 0.3436 and balanced accuracy was 0.5671. Held-out-machine test support was none 56,180, comp1 272, comp2 392, comp3 312, comp4 224, and multi_failure 80. Multi-failure precision and recall were zero. This is a separate generalization experiment, not a replacement for the chronological benchmark.

## Repeated-seed stability

For the selected 24h MLP/capped20 configuration, two-epoch CPU runs used seeds 7, 42, and 99. Test macro F1 was 0.3644, 0.3286, and 0.3373 respectively: mean 0.3434 and standard deviation approximately 0.0150. Test balanced accuracy was 0.5896, 0.6556, and 0.7519: mean 0.6657 and standard deviation approximately 0.0667. The spread, especially in balanced accuracy, shows that the compact run is not yet a stable production-quality estimate.

## Error analysis

For the selected chronological test run, the largest failure mode was `none` predicted as a component class: 1,812 as comp2, 879 as comp4, 690 as comp3, and 649 as comp1. The model predicted no multi-failure instances; actual multi-failures were most often predicted as comp2 (40), comp4 (15), comp1 (9), or comp3 (7). Component-class confusions were smaller than the none/failure confusion.

Top machine error rates in the selected test run were machine 63 (92/569, 16.17%), 52 (91/569, 15.99%), 94 (87/569, 15.29%), 67 (85/569, 14.94%), and 68 (85/569, 14.94%). These are diagnostic benchmark observations, not a basis for new test-derived features. Monthly error rates were 6.56% for 2015-10, 7.80% for 2015-11, and 7.67% for 2015-12. Rare-class support remains a material limitation, particularly for multi_failure.

## Tests

Focused Phase 6 tests passed: `8 passed in 6.03s` for architecture shapes, class weighting, threshold logic, calibration, and machine-held-out isolation. The full relevant Azure and regression selection passed `152 passed, 1 warning`; the warning was the pre-existing `PytestReturnNotNoneWarning` in `tests/test_influx_ingestion.py::test_config`.

The known `tests/test_training_contract.py` collection issue remains blocked because `training.train_multimodal` is absent. No replacement module was invented and the failure was not hidden.

## Artifacts and reproducibility

External artifacts include per-lookback architecture checkpoints, probability files, result JSON, temporal preprocessing state, baseline JSON, seed-stability checkpoints, the machine-held-out checkpoint, and `phase6_manifest.json`. The selected checkpoint metadata records architecture, label mapping, seed, epoch, validation score, and input metadata dimension. The manifest records source hashes, derived dataset hash, contract version, configuration, split context, thresholds, stability runs, and the explicit simulated-benchmark/no-performance-claim classification.

The model card and configuration artifacts are under the selected external output directory: `lookback_24h/mlp_capped20/model_card.md` and `lookback_24h/mlp_capped20/model_config.json`; preprocessing metadata is in the lookback directory. Their SHA-256 hashes are `9283BC049A2264442B1C06CB9FC4ECB503709E99136CCFC3754DAB21166F74C6` and `6CCEC7AD278D5E38003DF5FEBE355059052C765884B477A13715671CFAD46E7B`, respectively. All generated model artifacts and logs remain outside Git.

## Limitations and final status

- The two-epoch bounded runs are optimization evidence, not a final production training schedule.
- The neural model has zero multi-failure recall on the chronological test and held-out-machine test.
- Calibration is poor and the threshold experiment produced negligible improvement.
- The simple 24h logistic baseline outperformed the selected neural model on the principal test metrics.
- No random-forest or gradient-boosting experiment was run in this bounded package.
- No new test-derived features were introduced.
- Results describe a simulated benchmark only; no real-world factory performance is claimed.

Training and evaluation were authorized and did run for the bounded Phase 6 package. No raw data, acquisition manifest, FD001 artifact, existing IoT production module, or previous output was overwritten. No Git state-changing operation was performed. No additional model training or inference is implied beyond the experiments recorded in the external manifest.
