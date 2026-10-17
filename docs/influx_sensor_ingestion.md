# InfluxDB Sensor Ingestion

Part 5 consumes validated MQTT sensor messages and stores them in InfluxDB.
The ingester subscribes to `industrial/sensors` by default, validates every
payload with `iot.sensor_message.validate_sensor_message`, and writes one
InfluxDB point per approved FD001 cycle.

Stored points use measurement `fd001_sensor_reading` by default. `unit_id` is
stored as a tag, while `cycle` and the 17 selected FD001 sensor features are
stored as fields.

InfluxDB still requires a point timestamp for identity, but the FD001 `cycle`
is not a real measurement time. The ingester therefore uses a synthetic,
documented timestamp: fixed epoch `2000-01-01T00:00:00Z` plus one second for
each cycle index after cycle 1. This timestamp is only an idempotency key for
InfluxDB storage. It is derived only from the incoming unit/cycle identity,
does not use wall-clock ingestion time, and must not be interpreted as when the
engine reading happened.

## Dependencies

Install the runtime clients when running against real services:

```powershell
python -m pip install paho-mqtt influxdb-client
```

The existing local replay, schema, and unit tests keep these imports lazy, so
they can still run without those packages installed.

## Environment Variables

InfluxDB settings:

- `IOT_INFLUXDB_URL`: InfluxDB URL, for example `http://localhost:8086`.
- `IOT_INFLUXDB_ORG`: InfluxDB organization.
- `IOT_INFLUXDB_BUCKET`: target bucket.
- `IOT_INFLUXDB_TOKEN`: token with write access to the bucket. The documented
  `--query` helper also needs read access to that bucket.
- `IOT_INFLUXDB_MEASUREMENT`: optional measurement override; defaults to
  `fd001_sensor_reading`.

MQTT settings:

- `IOT_MQTT_HOST`: broker host; defaults to `localhost`.
- `IOT_MQTT_PORT`: broker port; defaults to `1883`.
- `IOT_MQTT_TOPIC`: topic; defaults to `industrial/sensors`.
- `IOT_MQTT_QOS`: QoS; defaults to `0`.

Do not commit tokens. The ingester validates that a token is present and avoids
printing raw token values in client-creation error messages.

## Local Service Setup

The following commands match the local Windows Docker workflow. Run them from
PowerShell. The named containers and persistent InfluxDB volume make the
service names and stored data predictable across restarts.

Start InfluxDB:

```powershell
docker run --name influxdb -p 8086:8086 -v influxdb2-data:/var/lib/influxdb2 influxdb:2
```

Open `http://localhost:8086` and complete the first-run setup with
organization `IOT` and bucket `industrial_sensors`. Create a token with read
and write access to that bucket.

Start Mosquitto with the existing local-development no-auth command:

```powershell
docker run --name mosquitto -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf
```

No-auth Mosquitto is suitable only for local development. Do not expose port
1883 or this broker to an untrusted network; use authentication, authorization,
and transport security for any shared or deployed environment.

Set the environment variables in the terminal that will run the ingester. The
following PowerShell prompt masks the token while it is entered and does not
print it or place it in the command history. The environment variable remains
available to child processes, so clear it when finished.

```powershell
$secureToken = Read-Host -Prompt "InfluxDB token" -AsSecureString
$tokenPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try { $env:IOT_INFLUXDB_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPtr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPtr) }
$env:IOT_INFLUXDB_URL = "http://localhost:8086"
$env:IOT_INFLUXDB_ORG = "IOT"
$env:IOT_INFLUXDB_BUCKET = "industrial_sensors"
$env:IOT_MQTT_HOST = "localhost"
$env:IOT_MQTT_PORT = "1883"
$env:IOT_MQTT_TOPIC = "industrial/sensors"
```

Do not replace the prompt with a real token in a script, commit the token, or
print the environment variable. When finished, clear it with
`Remove-Item Env:IOT_INFLUXDB_TOKEN`.

## Start Ingestion

Run the ingester in one terminal:

```powershell
python -m iot.influx_ingestion
```

Publish validated replay messages from another terminal:

```powershell
python -m iot.mqtt_publisher --topic industrial/sensors --limit 5
```

The publisher and ingester both use the existing sensor schema. Payloads with
extra fields such as `timestamp`, `rul`, or `failure_label` are rejected before
they can be written.

CLI arguments can override non-secret environment variables. Keep the token in
`IOT_INFLUXDB_TOKEN` rather than passing it on the command line:

```powershell
python -m iot.influx_ingestion `
  --influx-url http://localhost:8086 `
  --influx-org IOT `
  --influx-bucket industrial_sensors `
  --mqtt-host localhost `
  --mqtt-port 1883 `
  --mqtt-topic industrial/sensors
```

## Query Stored Records

Use the built-in query helper to fetch recent stored records as JSON:

```powershell
python -m iot.influx_ingestion --query --query-limit 5
```

Filter by `unit_id` tag:

```powershell
python -m iot.influx_ingestion --query --query-unit-id 1 --query-limit 5
```

Equivalent Flux query:

```flux
from(bucket: "industrial_sensors")
  |> range(start: 2000-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "fd001_sensor_reading")
  |> filter(fn: (r) => r.unit_id == "1")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 5)
```

Expected records include `unit_id`, `cycle`, and the selected features in
`iot.sensor_message.FD001_SENSOR_FEATURES`. The `_time` column is the synthetic
cycle identity timestamp described above, not a real event timestamp.

## Frozen LSTM inference

The queried JSON records can be passed to the frozen Sensor-only Baseline v1:

```powershell
python -m iot.lstm_inference .\queried_records.json
```

The JSON file must contain pivoted records for exactly one `unit_id`. The
adapter ignores Influx metadata (`_measurement`, `_time`, `_start`, `_stop`,
and `result`), validates the 17 approved features, sorts by `cycle`, requires
consecutive cycles, and uses the latest chronological 30-cycle window. It
returns one RUL prediction without requiring RUL or failure labels.

Inference uses these existing, frozen artifacts and never fits a scaler:

- `artifacts/data/fd001_lstm_baseline/scaler.joblib`
- `artifacts/models/fd001_lstm_baseline/best_model.pt`

The model input is always `float32` with shape `(1, 30, 17)` and uses the
feature order defined by `preprocessing.cmapss_fd001.SELECTED_FEATURES`.
The Influx `_time` value remains only a synthetic cycle identity timestamp; it
is not interpreted as the physical measurement time.
