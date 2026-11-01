# Phase 8–9 Final Azure PdM Integration Report

## Executive verdict

**READY FOR A COLLEGE DEMONSTRATION — OFFLINE REPLAY PATH.**

The end-to-end offline path works on Windows CPU with the real Phase 7 tuned Extra Trees artifact. It reads the acquired telemetry CSV, validates machine metadata and timezone-naive finite telemetry, accumulates contiguous 24-hour windows, applies the saved preprocessing and engineered feature order, performs threshold-adjusted six-class inference, and writes structured predictions with probabilities, provenance, benchmark status, and latency.

Live MQTT/InfluxDB integration is **not claimed**. The existing IoT components were inspected, but connecting the Azure benchmark to live services would require a separate approved adapter contract and service configuration. The guaranteed demonstration is deterministic local replay.

## Baseline and safety verification

Verified before and after implementation:

- Repository: `${IOT_FUSION_REPO_ROOT}`
- Branch: `feature/jacil-fusion-integration`
- HEAD: `1b6770f106f34dae5b52e225fbef8813d02f33fc`
- Git state-changing operations: none
- Raw CSVs and acquisition manifest: unchanged; hashes still match the manifest
- Phase 4 derived dataset hash: `431af6f141d559f30b64c9b0ec3295b2429c11b7a6188c6aca26398f6c60caf9`
- Phase 5 checkpoints: unchanged
- Phase 6/7 outputs: unchanged
- FD001 artifacts and existing IoT/NLP/fusion modules: not modified
- Existing untracked files: preserved

The selected Phase 7 model is:

`${AZURE_PDM_ROOT}\phase7_outputs\selected_extra_trees\model_tuned.pkl`

Its preprocessing, feature schema, thresholds, model card, and configuration are referenced by the Phase 7 v3 manifest. No preprocessing is refit during replay.

## What works end-to-end

The new isolated path consists of:

- `experiments/azure_pdm/streaming.py`: telemetry schema validation, machine metadata loading, duplicate detection, gap reset, incomplete-window accounting, latency measurement, and replay orchestration.
- `experiments/azure_pdm/demo.py`: Windows-friendly CLI.
- Existing `experiments/azure_pdm/inference.py`: saved artifact loading, exact input validation, preprocessing, feature construction, threshold-adjusted prediction, probabilities, model version, and provenance.
- External output: `${AZURE_PDM_ROOT}\phase8_9_outputs\offline_replay_predictions.jsonl`.

The executed replay used the real raw telemetry and machine metadata with the real tuned model, emitted 10 predictions, and produced:

| Measure | Result |
|---|---:|
| Records seen | 33 |
| Predictions emitted | 10 |
| Incomplete windows | 23 |
| Duplicate records rejected | 0 |
| Gap resets | 0 |
| Invalid records | 0 |
| Unknown machines | 0 |
| Mean inference latency | 15.42 ms |
| Maximum inference latency | 31.75 ms |

The first prediction used machine 1, observation time `2015-01-02 05:00:00`, and a contiguous input window from `2015-01-01 06:00:00` through `2015-01-02 05:00:00` as supplied to the model interface. Output explicitly states `simulated benchmark; offline replay; not a verified real-world failure`.

## Exact Windows PowerShell command

```powershell
Set-Location $env:IOT_FUSION_REPO_ROOT
$azureRoot = $env:AZURE_PDM_ROOT
if (-not $azureRoot) { throw 'Set AZURE_PDM_ROOT to the external Azure PdM data directory.' }
$demoRoot = Join-Path $azureRoot 'phase8_9_outputs'
New-Item -ItemType Directory -Path $demoRoot -Force | Out-Null

python -m experiments.azure_pdm.demo `
  --telemetry (Join-Path $azureRoot 'raw_acquisition_02\PdM_telemetry.csv') `
  --machines (Join-Path $azureRoot 'raw_acquisition_02\PdM_machines.csv') `
  --model (Join-Path $azureRoot 'phase7_outputs\selected_extra_trees\model_tuned.pkl') `
  --output (Join-Path $demoRoot 'offline_replay_predictions.jsonl') `
  --max-predictions 10 `
  --log-level INFO
```

The concise quickstart is [phase8_9_windows_quickstart.md](../docs/phase8_9_windows_quickstart.md), and the presentation runbook is [phase8_9_demo_runbook.md](../docs/phase8_9_demo_runbook.md).

## Architecture and data-flow boundary

The Mermaid diagram is [phase8_9_architecture.md](../docs/phase8_9_architecture.md). In summary:

1. Raw telemetry observations enter offline replay.
2. Exact schema, finite values, positive machine ID, timezone-naive timestamps, duplicates, and hourly continuity are checked.
3. Approved static machine metadata is joined by machine ID.
4. A complete contiguous 24-hour window is passed to saved train-fitted preprocessing.
5. Engineered statistics are generated in the frozen Phase 7 order.
6. The saved Extra Trees model returns probabilities.
7. Validation-frozen thresholds determine the output class.
8. The result contains probabilities, class, observation time, model version, provenance, latency, and benchmark warning.

Failure rows never enter runtime features. Errors and maintenance remain excluded because availability is unknown. They are not converted to text, and no technician narrative or BERT/NLP claim is made.

## Artifact acceptance and failure behavior

The inference layer rejects missing or extra fields, malformed timestamps, timezone-aware timestamps, non-finite sensor values, invalid machine metadata, non-contiguous windows, and incomplete windows. The streaming adapter rejects unknown machine IDs, rejects duplicate `(machineID, timestamp)` records, resets a machine buffer after a gap or out-of-order record, and never emits a prediction from an incomplete window.

The adapter checks the saved model lookback against the configured 24-hour window. Probability order is fixed as `none, comp1, comp2, comp3, comp4, multi_failure`; threshold-adjusted predictions use the thresholds recorded in the model artifact. No hardcoded prediction is used.

## Multimodal and IoT boundary

The existing FD001 fusion path expects a different RUL/regression contract and remains isolated. Existing NLP/fusion fixtures can demonstrate interface-level behavior, but they do not provide genuine Azure technician narratives and are not fused into this classifier. No 64-dimensional FD001 sensor embedding or 768-dimensional text embedding is connected to the Azure model.

The adapter is intentionally transport-neutral. Existing MQTT and InfluxDB modules were not modified or connected to live services. A future live adapter may feed validated records into `AzureTelemetryWindowAdapter`, but it must preserve the same timestamp, availability, schema, and provenance rules.

## Tests

Final relevant test selection:

```text
166 passed, 1 warning
```

The warning is the pre-existing `PytestReturnNotNoneWarning` in `tests/test_influx_ingestion.py::test_config`.

The new integration tests cover complete-window emission, duplicate rejection, gap reset, unknown machines, timezone-aware timestamps, non-finite values, deterministic inference, batch behavior, and invalid input rejection. The known `tests/test_training_contract.py` collection problem remains blocked because `training.train_multimodal` is missing; no artificial replacement was created.

## Deliverables

Repository additions:

- `experiments/azure_pdm/streaming.py`
- `experiments/azure_pdm/demo.py`
- `experiments/azure_pdm/tests/test_phase8_9_integration.py`
- `experiments/azure_pdm/docs/phase8_9_windows_quickstart.md`
- `experiments/azure_pdm/docs/phase8_9_demo_runbook.md`
- `experiments/azure_pdm/docs/phase8_9_architecture.md`
- `experiments/azure_pdm/docs/phase8_9_model_card.md`
- `experiments/azure_pdm/reports/phase8_9_final_integration_report.md`

External outputs:

- `${AZURE_PDM_ROOT}\phase8_9_outputs\offline_replay_predictions.jsonl`
- `${AZURE_PDM_ROOT}\phase8_9_outputs\phase8_9_reproducibility_manifest.json`

The reproducibility manifest records raw input hashes, acquisition manifest hash, selected model artifact hashes, Phase 7 manifest hash, replay output hashes, and hashes for the new code/documentation files. Its own SHA-256 at creation was `1c76f8ddab1ff531861fbd955cd3d0a10635ec89d2730d52344e451033cd05d4`.

## Final acceptance checklist

| Component | Status | Evidence |
|---|---|---|
| Saved Phase 7 artifact loading | Working | Real `model_tuned.pkl` loaded |
| Exact preprocessing and feature order | Working | Saved artifact state used; no refit |
| Threshold-adjusted six-class output | Working | Validation-frozen thresholds applied |
| Offline replay | Working | 10 real predictions emitted |
| Duplicate/gap/incomplete handling | Working | Unit tests and replay counters |
| Deterministic batch inference | Working | Integration tests and smoke test |
| Windows CPU launch path | Working | Quickstart command executed |
| Structured logs and latency | Working | Replay summary and per-prediction latency |
| Live MQTT/InfluxDB | Partially working | Transport-neutral adapter only; no live service connection |
| NLP/technician narrative fusion | Blocked/not applicable | No genuine narratives; availability policy excludes events |
| FD001 fusion reuse | Blocked by contract boundary | Different RUL/regression interface; kept isolated |
| `training.train_multimodal` suite | Blocked | Existing missing-module collection error |

The project is ready for a college presentation using the offline replay demonstration, provided the presentation clearly labels the dataset and predictions as simulated benchmark outputs and does not imply live cloud ingestion or verified real-world failures.
