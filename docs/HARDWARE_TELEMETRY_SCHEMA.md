# Hardware telemetry schema

Hardware telemetry uses schema version `1.0`, independent of the FD001 replay
contract. Its default MQTT topic is `maintainiq/hardware/telemetry`; publishers
send one JSON object every 5 seconds.

```json
{"schema_version":"1.0","device_id":"ESP32_001","session_id":"A1B2C3-unique","timestamp":"2026-11-20T10:30:00Z","sequence_number":1,"temperature":45.6,"vibration":1.3,"rpm":1500,"pressure":3.2,"operating_status":"normal"}
```

| Field | Type | Meaning and validation |
|---|---|---|
| `schema_version` | string | Exactly `1.0` |
| `device_id` | string | Required stable device identifier, up to 64 characters |
| `session_id` | string | Unique boot/session identifier; scopes sequence reset and deduplication |
| `timestamp` | string | ISO 8601 UTC; `Z` or zero offset; physical event time |
| `sequence_number` | integer | 1 through 2,147,483,647, increases each publish within a session |
| `temperature` | number | Celsius, -40 through 150 |
| `vibration` | number | mm/s, 0 through 50 |
| `rpm` | integer | revolutions/minute, 0 through 10,000 |
| `pressure` | number | bar, 0 through 20 |
| `operating_status` | string | `normal`, `warming`, `abnormal`, `failure`, or `recovery` |

The contract adds `session_id` to the proposed example so a reboot can start a
new sequence without being mistaken for a duplicate. The status enum names the
five simulator stages. Range limits and sample values are demonstration bounds,
not validated engineering thresholds. Unknown/missing fields, non-finite
values, wrong types, unsupported versions, malformed JSON, and non-UTC times
are rejected.

Deduplication is by `(device_id, session_id, sequence_number)`. The Python
ingester drops duplicate and out-of-order sequence values within its current
process and advances the watermark only after a successful Influx write. The
watermark is in memory and resets on service restart; Influx record identity
also includes device/session tags and event timestamp. QoS 1 may redeliver,
which is expected and handled. The event timestamp is never replaced with an
FD001 synthetic cycle time.
