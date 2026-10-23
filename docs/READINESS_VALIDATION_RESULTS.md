# Readiness Validation Results

**Date:** 2026-10-23
**Branch:** `feature/rafaz-readiness-remediation`
**Source revision:** `1ffc5be56b697a6fe3c97df0c32c1b5ed0f5bf3f`
**Environment:** Python 3.11.9; pytest 9.1.1
**Scope:** Focused tests only; not a full-suite run.

## Command scope

The focused pytest run targeted:

- `tests/test_fusion_alignment.py`
- `tests/test_missing_text.py`
- `tests/test_fixture_artifact_compatibility.py`
- `tests/test_sensor_message.py`
- `tests/test_local_replay.py`
- `tests/test_mqtt_publisher.py`
- `tests/test_influx_ingestion.py`
- `tests/test_lstm_inference.py`

## Observed result

- Passed: 92
- Failed: 1
- Passed subtests: 25
- Warnings: 2
- Duration: 58.71 seconds
- Pytest exit code: 1

**Overall status: BLOCKED / NOT ALL SELECTED TESTS PASS.**

## Failure

`TestLSTMInference.test_frozen_artifacts_predict_without_live_services` failed because the required checkpoint was absent:

`artifacts/models/fd001_lstm_baseline/best_model.pt`

The failure means this test did not verify prediction using the frozen checkpoint. The checkpoint's absence was already identified during the read-only repository audit.

**Do not regenerate data, retrain the model, or create a replacement checkpoint to clear this failure.** The baseline owner must locate and verify the authorized matching artifacts and their provenance.

## Warnings

1. `PytestReturnNotNoneWarning`: `tests/test_influx_ingestion.py::test_config` returned an `InfluxIngestionConfig` object instead of `None`.
2. `InconsistentVersionWarning`: a serialized scikit-learn `StandardScaler` was saved under scikit-learn 1.9.0 and loaded under 1.9.1. This compatibility warning requires follow-up; it is not proof that the artifact is safe or reproducible.

## Interpretation and limits

- These results apply only to the eight named test files at the recorded source revision and environment.
- They do not constitute a full-suite result or live broker, database, or hardware end-to-end validation.
- No fusion training, predictive-performance evaluation, official FD001 test evaluation, or baseline regeneration was performed by this test command.
- The missing checkpoint, dependency reproducibility, production provenance, genuine sensor/log pairing, and required data/evaluation approvals remain unresolved readiness gates.
