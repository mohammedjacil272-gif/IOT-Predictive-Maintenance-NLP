# Phase 7 — Azure PdM reliable prediction and inference

## Executive summary

Phase 7 produced a validation-selected improvement over the previous 24-hour logistic baseline while preserving all prior outputs. The selected model is an Extra Trees classifier using leakage-safe engineered telemetry aggregates and machine metadata. On validation it achieved macro F1 **0.4032**, versus **0.3835** for the prior 24-hour balanced logistic representation. After thresholds were frozen from validation, the final chronological test macro F1 was **0.4632**, balanced accuracy **0.7873**, and multi-failure precision/recall **0.340/0.708**. Standard argmax test results are also retained: macro F1 **0.3854**, balanced accuracy **0.5692**, and multi-failure precision/recall **0.500/0.014**.

The model and inference artifacts are for the simulated benchmark only. No real-world factory performance claim is made.

## Verification and provenance

Initial and final repository state remained:

- Branch: `feature/jacil-fusion-integration`
- HEAD: `1b6770f106f34dae5b52e225fbef8813d02f33fc`
- Existing untracked files were preserved.
- No Git state-changing operation occurred.

The canonical Phase 4 v3 dataset SHA-256 remained `431af6f141d559f30b64c9b0ec3295b2429c11b7a6188c6aca26398f6c60caf9`. All five raw CSV hashes continued to match the acquisition manifest. Existing Phase 5 checkpoint hashes remained unchanged, as did Phase 6 outputs and checkpoints.

Phase 7 artifacts are external under:

`${AZURE_PDM_ROOT}\phase7_outputs`

Final artifact manifest:

`${AZURE_PDM_ROOT}\phase7_outputs\phase7_manifest_final_v3.json`

## Independent metric verification

The saved Phase 6 probability array was independently recomputed with the repository metric implementation. The recomputed selected neural test macro F1 was 0.3286394955, balanced accuracy 0.6556224601, and its confusion matrix exactly matched the stored manifest.

The apparent 24-hour logistic discrepancy is not a label-order bug. Balanced accuracy is the arithmetic mean of the six class recalls; macro F1 is the arithmetic mean of the six class F1 values. The logistic model has high recall across rare classes, but low precision because many `none` rows are classified as failures. Therefore balanced accuracy can be 0.8986 while macro F1 is only 0.3792. All metric calculations use the explicit class order `none, comp1, comp2, comp3, comp4, multi_failure`, include unsupported classes with zero-valued precision/recall/F1, and use matching confusion-matrix labels.

The Phase 6 threshold search used validation probabilities only. The Phase 6 calibration statistic was computed from the untouched test probabilities; it was reported diagnostically and was not used for model selection. Machine-held-out preprocessing and model fitting used disjoint complete machine groups.

## Feature experiments

Features used only approved telemetry fields and machine metadata. The engineered representation contains, in stable sensor order, for each of `volt`, `rotate`, `pressure`, and `vibration`:

`mean`, `std`, `min`, `max`, `range`, `first_last_delta`, `slope`, and `mean_abs_diff`, followed by standardized machine age and train-vocabulary machine-model indicators.

All telemetry normalization and metadata vocabulary fitting used training rows only. No failure, error, maintenance, timestamp, split, sample ID, or target-provenance field was used as a feature. No interpolation or forward fill was introduced.

Validation macro F1 results:

| Lookback | Representation | Metadata | Validation macro F1 | Validation balanced accuracy | Multi-failure recall |
|---|---|---:|---:|---:|---:|
| 3h | mean/std | no | 0.2094 | 0.7014 | 0.3281 |
| 3h | mean/std | yes | 0.2223 | 0.7244 | 0.4063 |
| 3h | engineered | no | 0.2082 | 0.6948 | 0.2969 |
| 3h | engineered | yes | 0.2195 | 0.7163 | 0.3594 |
| 24h | mean/std | no | 0.3666 | 0.8946 | 0.8281 |
| 24h | mean/std | yes | 0.3835 | 0.8943 | 0.8125 |
| 24h | engineered | no | 0.3788 | 0.8828 | 0.7500 |
| 24h | engineered | yes | **0.3978** | 0.8875 | 0.7656 |

An installed Extra Trees candidate using the 24-hour engineered-plus-metadata representation achieved validation macro F1 **0.4032**, the highest validation score in the bounded comparison. It was selected without test inspection. Its validation balanced accuracy was 0.5893 and validation multi-failure recall was zero, illustrating why macro F1 and class-specific metrics were both retained.

## Rare-class and multi-failure findings

The Extra Trees model's standard chronological test confusion matrix was:

```text
[[53727, 737, 756, 283, 373, 0],
 [102, 130, 0, 0, 0, 0],
 [158, 0, 186, 0, 0, 0],
 [36, 0, 0, 132, 0, 0],
 [92, 0, 0, 0, 115, 1],
 [71, 0, 0, 0, 0, 1]]
```

Under argmax, multi-failure detection was poor: 1 of 72 multi-failure rows was recovered. Validation-only class thresholds materially changed the operating point. The frozen thresholds were `[1.0, 0.4, 0.4, 0.4, 0.4, 0.1]`; tuned validation macro F1 was 0.4231 and tuned chronological test macro F1 was 0.4632, with multi-failure precision 0.340 and recall 0.708. This is a decision-policy result, not evidence that the underlying class is reliably learnable in deployment.

The source class is rare: the canonical dataset contains 668 multi-failure rows overall and 72 in the chronological test. The class must therefore remain explicitly reported with support counts. Threshold tuning improves recall at a substantial precision trade-off and must be chosen for the intended operating objective using validation data only.

## Calibration and generalization

For the selected Extra Trees model, expected calibration error was 0.0743 on validation and 0.0753 on the standard test. These values are substantially lower than the Phase 6 neural ECE of 0.6232, but no separate post-hoc calibrator was adopted. The saved probabilities remain the model's native probabilities; thresholds are a separate decision configuration.

The complete-machine held-out experiment used machines 1–60 for training, 61–80 for validation, and 81–100 for test. Standard held-out test macro F1 was **0.4070**, balanced accuracy **0.5175**, and multi-failure precision/recall **0.571/0.050**. Thresholds tuned on the machine-held-out validation subset gave held-out test macro F1 **0.4596**, balanced accuracy **0.6956**, and multi-failure precision/recall **0.270/0.600**. These results are separate from the chronological benchmark and do not establish real-world generalization.

Global Extra Trees feature importance was saved in `selected_extra_trees/feature_importance.json`. The largest values were `volt_std`, `volt_mean`, `volt_max`, `volt_min`, and rotate change/range statistics. These are model-derived global importance values, not causal explanations.

## Inference interface

The reusable interface is:

`experiments/azure_pdm/inference.py`

It loads the serialized model and preprocessing state, validates exact telemetry fields, rejects timezone-aware timestamps, rejects malformed/non-finite values, requires a contiguous hourly window of the configured length, validates machine metadata, applies train-fitted normalization and model vocabulary, supports batch prediction, returns six-class probabilities, predicted class, model version, and provenance, and remains independent of MQTT, InfluxDB, and production IoT modules.

Final tuned inference artifact:

`${AZURE_PDM_ROOT}\phase7_outputs\selected_extra_trees\model_tuned.pkl`

The artifact and all associated schema, preprocessing, threshold, evaluation, feature-importance, model-card, configuration, and determinism files are recorded in `phase7_manifest_final_v3.json` with SHA-256 hashes. A real source telemetry window was passed through the public interface twice and produced identical results; batch prediction returned the same result.

## Tests

Phase 7 focused tests passed: **8 passed**. They cover finite/stable feature construction, model loading, deterministic single and batch inference, six-class output fields, and rejection of gaps and malformed values.

The broader Azure and relevant NLP/fusion/IoT regression selection previously passed **160 tests with one pre-existing warning**. The known `tests/test_training_contract.py` collection issue remains blocked because `training.train_multimodal` is missing; no artificial replacement was created.

## Files and artifacts

New repository files:

- `experiments/azure_pdm/feature_engineering.py`
- `experiments/azure_pdm/inference.py`
- `experiments/azure_pdm/phase7_runner.py`
- `experiments/azure_pdm/tests/test_phase7.py`
- `experiments/azure_pdm/reports/phase7_reliable_prediction_report.md`

External outputs include the validation sweep, selected Extra Trees and logistic artifacts, preprocessing and feature schemas, frozen threshold configuration, calibration summaries, confusion matrices, feature importance, machine-held-out results, determinism checks, and manifests.

No raw dataset, acquisition manifest, Phase 5 output, Phase 6 output, FD001 artifact, existing production module, or Git state was modified. No fabricated data, synthetic narratives, embeddings, live-system integration, or real-world performance claim was introduced.
