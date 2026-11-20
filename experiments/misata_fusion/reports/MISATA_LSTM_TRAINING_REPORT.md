# Misata Synthetic Industrial Dataset — LSTM Training Report

**Project:** IOT Predictive Maintenance NLP
**Experiment:** misata_fusion
**Experiment type:** Synthetic sensor-only predictive maintenance
**Training status:** COMPLETED
**Industrial deployment validation:** NOT VALIDATED
**Dataset:** sensor_readings.csv
**Dataset SHA256:** 9C7B6551E797A4CAF28AAE4399F3E1930BBA7BE0E5040E8F15A19BA5EA84AFDD

---

## 1. Executive Summary

A sensor-only LSTM binary classification experiment was completed using the synthetic Misata industrial sensor dataset.

The model predicts whether a failure will occur within the next 24 hours using historical sensor readings.

The experiment used chronological train, validation, and test partitions, training-only imputation and scaling, 30-hour input windows, and a 24-hour endpoint purge policy.

Training completed on an NVIDIA GeForce RTX 2050 GPU. The best checkpoint was selected using validation average precision. The classification threshold was selected using validation F1, then applied to the test set.

These results are exploratory results on synthetic data. They do not establish real-world industrial performance or deployment readiness.

## 2. Dataset and Input Features

Dataset characteristics:

- 52,560 hourly readings.
- Six industrial assets.
- One year of synthetic sensor data.
- 39 failure events.
- 63 structured work orders.

The model uses five sensor features:

1. vibration_mm_s
2. bearing_temp_c
3. motor_current_a
4. discharge_pressure_bar
5. rotor_speed_rpm

Target:

- `failure_next_24h`

The original dataset labels were preserved.

The following fields were excluded from model inputs to prevent leakage:

- degradation_stage
- rul_hours
- next_failure_mode
- failure_next_24h
- Future failure-event information
- Post-failure corrective work-order information

The dataset is synthetic. The experiment does not use genuine paired technician narratives.

## 3. Preprocessing

Preprocessing configuration:

| Setting | Value |
|---|---|
| Input window | 30 hourly readings |
| Prediction horizon | 24 hours |
| Sensor features | 5 |
| Prediction endpoints | Running readings only |
| Imputation | Median |
| Scaling | StandardScaler |
| Imputer fit | Training data only |
| Scaler fit | Training data only |
| Split strategy | Chronological |
| Random seed | 42 |

The final saved arrays were independently verified:

| Split | Input shape | Positive | Negative |
|---|---|---:|---:|
| Train | (25410, 30, 5) | 460 | 24,950 |
| Validation | (12957, 30, 5) | 230 | 12,727 |
| Test | (12979, 30, 5) | 207 | 12,772 |

All saved arrays passed the finite-value checks. The source dataset hash matched the preprocessing metadata.

## 4. Model Architecture and Training Configuration

Model: PyTorch LSTM binary classifier.

| Parameter | Configuration |
|---|---|
| Architecture | Two-layer LSTM |
| Input size | 5 |
| Hidden size | 64 |
| Dropout | 0.2 |
| Output | One binary classification logit |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Batch size | 64 |
| Maximum epochs | 40 |
| Early stopping patience | 7 |
| Random seed | 42 |
| Loss | BCEWithLogitsLoss |
| Positive class weighting | 54.2391 |
| Training device | NVIDIA GeForce RTX 2050 |
| Mixed precision | Disabled |

Class weighting was calculated from the training partition to account for the low failure prevalence.

Early stopping monitored validation average precision.

## 5. Training Results

Training stopped early at epoch 13.

The best checkpoint was selected at epoch 6.

| Epoch | Training loss | Validation AP |
|---:|---:|---:|
| 1 | 0.75496 | 0.09063 |
| 2 | 0.50163 | 0.11610 |
| 3 | 0.42560 | 0.09905 |
| 4 | 0.40521 | 0.10955 |
| 5 | 0.39746 | 0.12733 |
| 6 | 0.40163 | 0.18815 |
| 7 | 0.42558 | 0.10716 |
| 8 | 0.35961 | 0.15464 |
| 9 | 0.35306 | 0.10828 |
| 10 | 0.31495 | 0.09601 |
| 11 | 0.31525 | 0.10208 |
| 12 | 0.32499 | 0.10040 |
| 13 | 0.30277 | 0.09594 |

Best validation average precision: approximately 0.18815.

## 6. Validation Evaluation

The decision threshold was selected on the validation set by maximizing F1.

Selected threshold:

`0.9184601306915283`

| Metric | Validation result |
|---|---:|
| Average Precision (AP) | 0.18815 |
| ROC-AUC | 0.87901 |
| Precision | 0.21942 |
| Recall | 0.26522 |
| F1 | 0.24016 |

Confusion matrix, ordered as TN, FP, FN, TP:

`[12510, 217, 169, 61]`

## 7. Test Evaluation

The selected validation threshold was frozen and applied to the test set.

| Metric | Test result |
|---|---:|
| Average Precision (AP / PR-AUC) | 0.23194 |
| ROC-AUC | 0.91229 |
| Precision | 0.22785 |
| Recall | 0.26087 |
| F1 | 0.24324 |
| Threshold | 0.91846 |

Confusion matrix, ordered as TN, FP, FN, TP:

`[12589, 183, 153, 54]`

Test set:

- Total samples: 12,979
- Actual positives: 207
- Actual negatives: 12,772
- True positives detected: 54
- False negatives: 153
- False positives: 183
- True negatives: 12,589

The model detected 54 of the 207 positive test endpoints at the selected threshold. It missed 153 positive endpoints and generated 183 false positives.

The AP score is higher than the test-set positive prevalence, indicating useful ranking signal in this synthetic experiment. However, the selected operating threshold produces limited precision and recall.

## 8. Saved Artifacts

Output directory:

`experiments/misata_fusion/models/lstm_run_001/`

Artifacts:

- `best_model.pt` — selected model checkpoint.
- `config.json` — model and training configuration.
- `metrics.json` — validation and test metrics.
- `training_history.csv` — epoch-level training history.

Checkpoint SHA256:

`36972CDA9FC5A02FCB39944A0C57849C6A2F7A12AEADBDC3BE520CA8A5759ECB`

The metrics record identifies the dataset as synthetic and industrial deployment validation as false.

## 9. Interpretation and Limitations

The experiment establishes that the sensor-only LSTM training and evaluation pipeline completed successfully on the available synthetic dataset.

It does not establish:

- Generalization to real industrial telemetry.
- Reliability across unseen industrial sites or equipment types.
- Production-ready failure detection.
- Acceptable operational false-alarm rates.
- Validated sensor-text fusion.
- Performance using genuine technician narratives.

The model misses a substantial number of positive test endpoints at the selected threshold. Its operating characteristics require further investigation before any operational use.

The dataset contains approximately 1.6–1.8% positive running endpoints across the chronological splits, making class imbalance an important consideration.

Results should be described as an exploratory synthetic-data baseline, not as a validated industrial predictive-maintenance solution.

## 10. Recommended Next Steps

1. Preserve this training run and its checkpoint.
2. Reconcile the dataset audit with the completed training status.
3. Evaluate threshold trade-offs and calibration without changing the frozen test result.
4. Consider asset-wise and failure-mode-wise evaluation where sample counts permit.
5. Complete the technician-log NLP pipeline.
6. Obtain genuine time-aligned sensor and technician-text data.
7. Train and evaluate sensor-text fusion only when suitable paired data is available.
8. Recreate and verify the missing NASA C-MAPSS FD001 checkpoint.
9. Complete integration, regression, and clean Windows installation testing.
10. Prepare the final project report with synthetic and real-data limitations clearly separated.

---

**End of report.**
