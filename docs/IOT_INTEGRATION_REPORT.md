# IoT integration report

The local contracts support timezone-aware MQTT sensor messages, validation,
deterministic payload handling, and Influx point construction. Real event time
is preserved separately from ingestion time. FD001 replay may use an explicitly
marked synthetic cycle-derived timestamp; live mode rejects that fallback.

MQTT and database connectivity were not exercised against production services.
The repository contains dry-run/mocked verification only. Credentials remain
outside source code. A real deployment still requires an authorized broker,
InfluxDB bucket/retention policy, and operational observability.
