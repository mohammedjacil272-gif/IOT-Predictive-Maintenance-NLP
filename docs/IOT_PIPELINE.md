# IoT pipeline

Status: local replay and schema validation implemented; live infrastructure not
verified.

The safe local path is MQTT payload validation -> Influx point conversion ->
sensor inference. Real payloads may provide an offset-aware `event_timestamp`,
which is used as the Influx point time. Live callers can set
`InfluxIngestionConfig.allow_synthetic_timestamps=False` to reject cycle-only
messages. The FD001 replay has cycle indices only and therefore uses a
documented synthetic timestamp solely for idempotency.
Cycle must never be interpreted as wall-clock time.

MQTT and Influx dependencies are lazy and services are not contacted by default.
No production broker, database, or credentials were accessed in this phase.
