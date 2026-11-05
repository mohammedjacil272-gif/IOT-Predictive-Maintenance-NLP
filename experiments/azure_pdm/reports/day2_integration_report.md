# Day 2 Integration Report

## Release verdict

**READY for the college demonstration through the Azure offline replay path.**

The actual Phase 7 Azure artifact loads and runs end-to-end on real acquired telemetry. The FD001 checkpoint was recovered read-only from the original sibling worktree, but was not copied into this worktree. NLP rule-based components and synthetic/interface fusion contracts are working; transformer embeddings and genuine multimodal fusion remain explicitly blocked.

## FD001 asset recovery

The exact frozen checkpoint was found at:

`${FD001_RECOVERY_ROOT}\artifacts\models\fd001_lstm_baseline\best_model.pt`

SHA-256:

`388d6e45f6d9692aee1d8cc9d8de021c4fab4eb0ce8287445eb24973cc2db013`

The sibling worktree's `model_config.json`, `training_config.json`, and `split_config.json` hashes match the corresponding integration-worktree configuration files. A direct read-only smoke test using the current scaler and the recovered external checkpoint produced:

```text
unit_id=7, window=1..30, prediction=155.6051483154297, input_shape=(1, 30, 17)
```

The checkpoint was not copied, overwritten, regenerated, or substituted. The repository's fixed-path frozen-checkpoint regression remains blocked until the owner explicitly supplies or restores the artifact in this worktree.

## NLP dependency and fallback

Environment findings:

- Python: 3.11.9
- `transformers`: unavailable
- `paho-mqtt`: available
- `influxdb-client`: available
- Free disk observed: approximately 26.23 GB
- No verified local DistilBERT model/tokenizer artifact was available.
- No model download was attempted.

`nlp/embeddings.py` now lazy-loads the optional dependency and requests local-only model artifacts. It returns a structured `embedding_diagnostics()` result and raises a clear `RuntimeError` when embeddings are unavailable. Rule-based preprocessing, NER, and severity remain usable offline. Empty, malformed, and deterministic fixture behavior is covered by tests. No genuine technician logs were created.

## Sensor and Azure integration

The FD001 sensor/MQTT/Influx contracts remain separate from Azure because FD001 uses `unit_id/cycle/17 features` and Azure uses `machineID/datetime/4 sensors`. Existing MQTT and Influx modules were exercised only through mocks; no live broker or database was contacted.

The Azure adapter uses the saved Phase 7 tuned Extra Trees artifact and validates:

- exact telemetry fields;
- positive machine identity and approved metadata;
- timezone-naive timestamps;
- finite numeric values;
- duplicate `(machineID, timestamp)` records;
- hourly continuity and gap resets;
- complete 24-hour windows;
- saved preprocessing and feature order;
- six probability classes and validation-frozen thresholds.

Bounded multi-machine replay used generated input derived from the raw CSV without modifying it:

| Measure | Result |
|---|---:|
| Machines | 1, 2, 3 |
| Records seen | 78 |
| Predictions emitted | 9 |
| Incomplete windows | 69 |
| Duplicates rejected | 0 |
| Gap resets | 0 |
| Invalid records | 0 |
| Unknown machines | 0 |
| Output | `phase8_9_outputs/multi_machine_predictions_09.jsonl` |

The output contains exactly three predictions per machine after each machine's first 24-hour history, with actual model probabilities and provenance. No hardcoded predictions were used.

The original single-machine replay also remains available at `phase8_9_outputs/offline_replay_predictions.jsonl`.

## Fusion and NLP boundary

The fusion contracts enforce canonical `(unit_id, cycle)` alignment, 64-D sensor embeddings, 768-D text embeddings, explicit namespaces, provenance, split isolation, missing-log policy, timestamp policy, and approval gates. The restored `training/train_multimodal.py` provides the existing fail-closed loader and dataset interface; its `main()` explicitly blocks unapproved training. It does not fabricate a model or generate artifacts.

Synthetic/interface fusion tests pass for wrong dimensions, missing modalities, missing text, duplicate mappings, provenance mismatches, split leakage, and invalid approvals. Genuine fusion is not implementable from current data because there are no approved aligned genuine sensor/text records, availability evidence, or authorized labels/training protocol.

The Azure structured error/maintenance tables remain excluded from runtime inputs. They are not technician narratives and no NLP claim is made for them.

## Repairs made

1. `scripts/verify_lstm_baseline.py`: read-only verification no longer generates missing baseline data and checks missing trained artifacts before data validation.
2. `training/train_multimodal.py`: restored the existing contract-compatible fail-closed loader/dataset interface; no training entry point was added.
3. `nlp/embeddings.py`: added lazy optional dependency loading, local-only model loading, diagnostics, and clear offline failure behavior.
4. Added `tests/test_nlp_embeddings_diagnostics.py`.
5. Added the Day 2 inventory and this report.

## Tests

Passing relevant suite:

```text
187 passed, 1 pre-existing warning
```

The warning is `PytestReturnNotNoneWarning` in `tests/test_influx_ingestion.py::test_config`.

Blocked separately:

```text
tests/test_lstm_inference.py::TestLSTMInference::test_frozen_artifacts_predict_without_live_services
```

Cause: the exact checkpoint is absent from the integration worktree. It was recovered externally and verified, but deliberately not copied. No checkpoint was regenerated.

## Protected verification

- Branch unchanged: `feature/jacil-fusion-integration`
- HEAD unchanged: `1b6770f106f34dae5b52e225fbef8813d02f33fc`
- Raw Azure CSV hashes still match `acquisition_manifest_02.json`.
- Phase 5 and Phase 7 checkpoint hashes remain unchanged.
- Phase 4–9 external output directories remain intact.
- No production MQTT/InfluxDB connection was made.
- No raw dataset, manifest, frozen checkpoint, model artifact, or existing Phase 4–9 output was modified.
- No commit, push, merge, rebase, reset, clean, stash, or branch switch occurred.

Updated inventory: [day2_component_inventory.json](day2_component_inventory.json).

Updated external reproducibility manifest: `${AZURE_PDM_ROOT}\phase8_9_outputs\phase8_9_reproducibility_manifest_v3.json` (SHA-256 `95d9ea1d5e4855d2253393d235b464deeaa287b2bf618356797401b6b8179372`).

## Final known limitations

- FD001 runtime regression requires owner-approved restoration of the recovered checkpoint into the expected local path.
- Transformer embeddings require a pinned `transformers` environment and approved local model/tokenizer artifacts.
- Fusion remains synthetic/interface-only; no performance result is claimed.
- Azure live MQTT/Influx transport is not connected.
- Azure outputs remain simulated-benchmark predictions, not verified real-world failures.

## Work assignment after independent branch reviews

### Rafaz

Review the FD001/IoT branch independently. Verify the recovered checkpoint lineage and decide whether the exact external `best_model.pt` may be restored into the expected local artifact path. Review MQTT topic/configuration and Influx schema compatibility. Do not retrain or replace the checkpoint. Return a hash-confirmed artifact decision and any transport defects.

### Wahid

Review the NLP/fusion branch independently. Verify the pinned transformer dependency/model-artifact plan, synthetic fixture provenance, 64-D/768-D alignment, missing-modality policies, and approval gates. Do not create genuine narratives or train fusion. Return a dependency decision and a contract-level fusion readiness assessment.

Both reviews must remain read-only with respect to raw data, frozen artifacts, and existing Phase 4–9 outputs until the team explicitly authorizes any restoration or dependency installation.
