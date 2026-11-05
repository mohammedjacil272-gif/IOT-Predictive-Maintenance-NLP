# IoT End-to-End Test Readiness

**Audit scope:** `1ffc5be56b697a6fe3c97df0c32c1b5ed0f5bf3f`. This document distinguishes source code and mock-based unit tests from live service and hardware validation.

## What exists in the audited tree

The audited `main` tree contains software components for validated sensor messages, deterministic local FD001 replay, MQTT publishing, MQTT-to-InfluxDB ingestion, and a frozen LSTM inference adapter. Tracked test files include:

- `tests/test_sensor_message.py`
- `tests/test_local_replay.py`
- `tests/test_mqtt_publisher.py`
- `tests/test_influx_ingestion.py`
- `tests/test_lstm_inference.py`

Reviewed MQTT/Influx tests use fake/mock clients and in-memory objects. That is software-level validation, not evidence of a real broker/storage round-trip, physical device integration, hardware-in-the-loop behavior, or production security. A fresh test run on the exact audit SHA must be recorded separately; live-service/HIL results are not established by this document.

## Prerequisites before genuine integration testing

- Correct source SHA and approved non-production test checkout.
- Isolated MQTT broker and InfluxDB endpoint/bucket.
- Least-privilege non-production credentials supplied through approved environment/secrets handling; never commit tokens or paste them into logs.
- Approved MQTT topic/QoS, payload schema, timestamp/timezone semantics, Influx measurement/tags/fields, retention, and cleanup procedure.
- Safe non-production messages and expected-result fixture.
- Approved dependency manifest and actual service client packages.
- Permission to use any data and clarity that official FD001 test labels or genuine technician text will not be sent into an unauthorized system.

Code references these environment variable names; values must be supplied by the owner, not invented:

- MQTT: `IOT_MQTT_HOST`, `IOT_MQTT_PORT`, `IOT_MQTT_TOPIC`, `IOT_MQTT_QOS`, `IOT_REPLAY_LIMIT`, `IOT_TRAIN_PATH`.
- InfluxDB: `IOT_INFLUXDB_URL`, `IOT_INFLUXDB_ORG`, `IOT_INFLUXDB_BUCKET`, `IOT_INFLUXDB_TOKEN`, `IOT_INFLUXDB_MEASUREMENT`.

Third-party clients are loaded lazily. Mock tests can pass without connecting to a service or even importing a live client package.

## Staged acceptance criteria — live evidence still pending

1. **Broker connectivity:** connect/authenticate, publish/subscribe, disconnect against isolated broker; record software and broker versions.
2. **Payload/schema:** valid payload round-trip; malformed, missing, wrong-typed and non-finite fields are rejected and counted.
3. **Storage round-trip:** write known rows to isolated InfluxDB, query back, verify counts, tags, fields, units and timestamps.
4. **Duplicates/order:** define duplicate and out-of-order semantics; replay duplicates and reordered messages and verify expected outcomes.
5. **Recovery:** controlled broker/database interruption and restart; measure losses/duplicates and confirm recovery behavior or document limits.
6. **Latency/capacity:** approve throughput, latency percentiles, test duration and failure budget before testing; publish measured results.
7. **Inference boundary:** verify no RUL labels, terminal/future cycles or official test targets enter live inference. Confirm frozen baseline input order/scaler/window/checkpoint identity without modifying them.
8. **Security:** ensure secrets are not logged, credentials are non-production and least-privilege, network access is restricted, transport security is configured per policy, retention/cleanup works.
9. **Target device:** claim hardware/device readiness only after the named target/gateway is tested with approved messages and saved expected-result evidence.

## Safe software-only command

After dependencies/runtime are approved, run from repository root:

```powershell
python -m pytest tests/test_sensor_message.py tests/test_local_replay.py tests/test_mqtt_publisher.py tests/test_influx_ingestion.py tests/test_lstm_inference.py -q
```

This runs software tests, mostly using mocks/fakes; it does not automatically run a genuine end-to-end service test. Do not claim live IoT readiness from these tests alone.
