# Current Phase and Readiness

## Current implementation

The repository has progressed beyond the original preprocessing-only phase. It contains FD001 sensor preprocessing and an LSTM baseline, technician-log NLP and embedding artifact generation, metadata-based fusion components, and IoT ingestion/replay code. Software tests cover selected interfaces; each recorded result applies only to its named source revision, environment, and command.

## Safe current work

- Maintain software contracts using synthetic fixtures and reviewable tests.
- Preserve namespace separation, record provenance, and entity-level `log_id` ownership.
- Inspect frozen baseline/generated artifacts read-only unless a separately approved task requires writing.
- Keep unverified or historical test results labeled with their exact scope.

## Blocked work

Genuine fusion training and performance evaluation remain blocked pending approved permissioned paired data, authoritative equipment mapping and shared cycle semantics, point-in-time availability/cutoff rules, target and horizon, equipment-level splits, missing/multiple-log decisions, embedding and baseline-checkpoint lineage, and explicit training authorization. Final held-out evaluation requires separate authorization.

The expected baseline checkpoint is absent from the checked-in inventory. Do not retrain the baseline, regenerate frozen artifacts, create substitute checkpoints, or use official FD001 test labels for training/model selection to clear this gap.

## IoT readiness

MQTT, InfluxDB, and replay software and mock/software tests exist. They do not establish live broker, database, device, or hardware validation. Such validation requires an approved isolated setup and recorded evidence.

See the [approval checklist](MULTIMODAL_APPROVAL_CHECKLIST.md), [decision register](../docs/MULTIMODAL_DECISION_REGISTER.md), [evaluation protocol](../docs/MULTIMODAL_EVALUATION_PROTOCOL.md), and [readiness results](../docs/READINESS_VALIDATION_RESULTS.md).
