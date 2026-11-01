# IoT End-to-End Test Readiness

## Existing tests and verified scope

No MQTT, InfluxDB, device, hardware-in-the-loop, or IoT end-to-end tests were
found. Existing automated tests are unit/contract tests for fusion tensor
shapes, metadata alignment, provenance, split checks, RUL handling, and
missing-text behavior. The fixture verification scripts validate five
synthetic log records and sensor vectors. They are saved-artifact software
checks only, not a live/replayed source, device, or hardware result.

The repository's architecture and methodology describe MQTT, InfluxDB, and an
IoT demonstration as planned later-phase work. This review found no IoT
transport/storage implementation or real hardware validation evidence.

## Remaining unverified work

- Telemetry acquisition, device identity, cycle/time semantics, ordering,
  duplicate delivery, missing readings, and clock behavior.
- MQTT topic/message schema, delivery guarantees, reconnect/error behavior,
  authentication, and transport security.
- InfluxDB schema, write/read behavior, retention, timestamp handling, and
  recovery from service interruption.
- End-to-end transformation from genuine sensor and permissioned technician
  sources through storage, feature/embedding generation, alignment, and model
  inference.
- Point-in-time log availability and prediction-cutoff enforcement in a live
  or replayed stream.
- Resource use, latency, restart/recovery behavior, and acceptance criteria on
  the intended deployment target.

## Inputs needed before hardware validation

The team must identify the target device or approved gateway/test rig, genuine
permissioned telemetry and log sources, an accessible MQTT broker and InfluxDB
environment (or explicitly approved equivalents), credentials/access controls,
message and timestamp contracts, and an approved test procedure with expected
results. No such hardware, service configuration, permissioned paired data, or
test results are established by the current synthetic fixture.

Do not report software fixture passes as IoT or hardware validation.

## Missing baseline checkpoint is a separate blocker

`scripts/verify_lstm_baseline.py` requires both `best_model.pt` and
`last_model.pt`; its read-only behavior reports missing files and does not
regenerate them. These files are absent from the checked-in baseline model
directory. The repository's `.gitignore` excludes `*.pt`, which is consistent
with checkpoints being distributed outside Git, but no delivery location or
owner confirmation is recorded. Checkpoint availability is therefore
**BLOCKED / awaiting the baseline owner**. Do not rerun baseline training to
make a verifier pass. This is separate from IoT hardware readiness and is not
evidence that any live or replay test has run.

