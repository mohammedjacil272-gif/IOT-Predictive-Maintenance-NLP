# Day 1 Technical Audit — IoT predictive maintenance, NLP, and fusion

## Executive status

The project now has a reliable, demonstrable Azure PdM offline inference path, while the FD001 sensor/MQTT/Influx and synthetic NLP/fusion paths remain correctly separated. The Azure path is suitable for a college demonstration. A live Azure MQTT/Influx integration and genuine multimodal fusion are not claimed.

## Verified repository state

- Repository: `${IOT_FUSION_REPO_ROOT}`
- Branch: `feature/jacil-fusion-integration`
- HEAD: `1b6770f106f34dae5b52e225fbef8813d02f33fc`
- Git state-changing operations: none
- Worktrees were inspected; no worktree was changed.
- Raw Azure datasets, acquisition manifest, Phase 4–9 external outputs, and FD001 artifacts were not modified.
- Existing untracked files were preserved.

The machine-readable inventory is [day1_component_inventory.json](day1_component_inventory.json).

## Architecture and dependency map

### FD001 sensor and IoT path

`iot/sensor_message.py` validates the frozen FD001 schema: positive `unit_id`, positive `cycle`, and 17 finite numeric features from `preprocessing/cmapss_fd001.py`. `iot/local_replay.py` parses FD001 training rows. `iot/mqtt_publisher.py` serializes validated messages to the default `industrial/sensors` topic and supports injected mock clients. `iot/influx_ingestion.py` validates incoming MQTT payloads, maps them to `fd001_sensor_reading`, and provides a query adapter. `iot/lstm_inference.py` consumes 30 consecutive cycles and the frozen scaler/checkpoint to produce one RUL value.

This path is internally contract-oriented and mock-tested, but its runtime frozen-checkpoint regression is blocked because `artifacts/models/fd001_lstm_baseline/best_model.pt` is absent. The checkpoint was not recreated or substituted.

### NLP path

`nlp/preprocessing.py` preserves original text and emits normalized text/status. `nlp/ner.py` emits deterministic rule-based entities with source offsets. `nlp/severity.py` emits heuristic severity and evidence. `nlp/embeddings.py` defines frozen DistilBERT mean-pooled 768-dimensional embeddings, but the `transformers` dependency is not installed in this environment and no genuine Azure narratives are available.

The NLP tests exercise malformed/empty text, NER offsets, and severity behavior. NLP artifacts are explicitly synthetic/interface fixtures, not genuine technician records.

### Fusion path

`training/fusion_dataset.py`, `training/fusion_provenance.py`, `training/fusion_splits.py`, `training/nlp_fusion_adapter.py`, and `training/nlp_fusion_pipeline.py` enforce canonical `(unit_id, cycle)` alignment, 64-dimensional sensor embeddings, 768-dimensional text embeddings, split isolation, namespace, provenance, point-in-time, and missing-log policies. `models/multimodal_fusion.py` is an RUL regression model, not an Azure six-class classifier.

The fusion path is implemented and contract-tested for synthetic/interface data. Genuine fusion remains blocked by approval, availability, source-lineage, and aligned-data requirements. The FD001 64-D sensor embedding and 768-D text embedding are intentionally not connected to the Azure Extra Trees model.

### Azure PdM path

`experiments/azure_pdm/feature_engineering.py` creates the frozen 24-hour telemetry aggregate schema. `inference.py` loads the Phase 7 tuned Extra Trees model, preprocessing, feature order, and validation-frozen thresholds. `streaming.py` validates machine metadata and timezone-naive hourly telemetry, rejects duplicates and malformed values, resets on gaps, waits for complete windows, and measures latency. `demo.py` provides the Windows CPU offline replay.

The verified path is:

```text
raw PdM_telemetry.csv
  -> schema/finite/time/duplicate/gap checks
  -> approved PdM_machines.csv metadata
  -> contiguous 24-hour window
  -> saved train-fitted preprocessing
  -> engineered telemetry features
  -> saved Phase 7 Extra Trees
  -> validation-frozen thresholds
  -> six-class prediction/probabilities/provenance/benchmark warning
```

Failure records remain target-only. Errors and maintenance remain excluded because availability is unknown. No NLP branch is implied for Azure structured events.

## Defects found and safe repairs

| Severity | Defect | Repair status |
|---|---|---|
| Critical | `training.train_multimodal` was missing although contract tests imported it and existing history contained its interface. | Repaired by restoring a fail-closed compatibility loader/dataset module. It does not train or generate artifacts; six contract tests pass. |
| High | `scripts/verify_lstm_baseline.py` generated missing baseline data during read-only verification. | Repaired: verification now fails closed without creating data. Missing trained artifacts are checked first with explicit “read-only; not regenerate” wording. |
| High/blocker | FD001 runtime inference test requires absent `best_model.pt`. | Not repaired by design. Checkpoint recreation/substitution is prohibited; test remains separately blocked. |
| Medium | Azure model had no transport-neutral integration into the existing IoT path. | Repaired safely with an isolated offline-first Azure telemetry adapter and CLI; no live service connection. |
| Medium | No explicit Azure replay tests for duplicate/gap/incomplete/timezone handling. | Added focused integration tests; they pass. |
| Low | Broad system does not have one production deployment contract joining FD001, NLP, and Azure. | Documented as an architecture boundary; no unsafe unification was attempted. |

## Sensor pipeline findings

FD001 sensor validation, MQTT serialization, publisher configuration, and Influx point/query adapters are implemented and mock-tested. Defaults are explicit: localhost MQTT, `industrial/sensors`, QoS 0, and `fd001_sensor_reading`. The FD001 cycle field is explicitly not a wall-clock timestamp; Influx uses a synthetic cycle-derived timestamp.

The Azure adapter does not reuse that FD001 schema because the schemas and timestamp semantics differ. It accepts `datetime`, `machineID`, `volt`, `rotate`, `pressure`, and `vibration`, plus approved machine `model` and `age`. It does not connect to production services.

## NLP and fusion findings

NLP preprocessing, NER, and severity are deterministic rule-based components over maintenance text. Embedding generation is external-model dependent and currently unavailable because `transformers` is not installed. The current fixtures are synthetic and explicitly namespaced. No genuine technician narrative claim is made.

Fusion alignment and provenance checks are stronger than the available data: they reject missing lineage, unauthorized equipment, official-test contamination, non-finite embeddings, duplicate mappings, namespace mismatch, missing point-in-time policies, and unapproved genuine training. This is appropriate. No fusion model was retrained because genuine aligned sensor/text data, availability semantics, target approvals, and training authorization do not exist.

## Test results

Passing relevant suite after repairs:

```text
185 passed, 1 pre-existing warning
```

The warning is `PytestReturnNotNoneWarning` from `tests/test_influx_ingestion.py::test_config`.

The blocked test is:

```text
tests/test_lstm_inference.py::TestLSTMInference::test_frozen_artifacts_predict_without_live_services
```

Reason: `artifacts/models/fd001_lstm_baseline/best_model.pt` is absent. It was not regenerated. All non-artifact-dependent LSTM tests pass.

The restored `tests/test_training_contract.py` collection now passes all 6 tests.

## Protected-artifact verification

The following known hashes remained unchanged:

- Phase 5 3h checkpoint: `7374989a5c28f8edaaecfdaa1fb309de8195538ddb061efbf9ddd4ac3977c554`
- Phase 5 24h checkpoint: `7b326c118e997ed82bbc53459bc98a7da764e6e65b55d598238ee8c1f00ce59e`
- Phase 7 tuned Azure model: `67e1188ea2d753f298658428d4cc3e588085b4425cf1f293ad06314fc92e462d`

Raw Azure hashes continued to match the acquisition manifest. No raw CSV, acquisition manifest, FD001 checkpoint, Phase 4–9 output, or production IoT/NLP/fusion module was altered.

## Day 2 implementation plan

1. **Presentation hardening:** run the Windows quickstart from a clean shell, capture one representative JSONL prediction and replay summary, and prepare screenshots/slide notes using only measured output.
2. **Optional transport adapter design:** if service credentials and explicit approval become available, add a separate Azure transport adapter that feeds the existing `AzureTelemetryWindowAdapter`; do not alter FD001 MQTT/Influx contracts.
3. **FD001 restoration decision:** obtain the owner-approved frozen checkpoint or formally mark FD001 runtime inference as unavailable. Do not retrain under deadline pressure.
4. **NLP dependency decision:** either pin/install the approved transformer artifacts in a reproducible environment or keep the NLP section interface-only and synthetic. Do not invent technician narratives.
5. **Fusion readiness gate:** keep fusion training blocked unless genuine aligned sensor/text records, availability timestamps, target definitions, split manifest, approvals, and artifact lineage are supplied.
6. **Final regression:** rerun the passing suite plus the offline Azure replay; report the FD001 checkpoint test separately as blocked if the protected artifact remains absent.

## Final status

Working: Azure offline replay, saved-model inference, threshold policy, schema/finite validation, metadata validation, duplicate/gap/incomplete handling, deterministic output, FD001 IoT mocks, NLP fixture processing, and fusion contract validation.

Partially integrated: FD001 MQTT/Influx are mock-tested but not connected to Azure inference; the Azure transport adapter is offline-only.

Blocked: FD001 frozen checkpoint runtime regression, live service integration without approval/configuration, transformer-backed embedding generation in this environment, genuine technician-narrative fusion, and multimodal training.
