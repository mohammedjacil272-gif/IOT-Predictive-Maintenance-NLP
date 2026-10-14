# Local Sensor Replay

Milestone 2 replays FD001 training rows locally as validated IoT sensor
messages. It reads `data/train_FD001.txt`, preserves `unit_id` and `cycle`, and
emits only the 17 selected sensor features accepted by `iot.sensor_message`.
It does not add RUL, failure labels, timestamps, official test targets, MQTT,
or InfluxDB dependencies.

Run a small replay as JSON lines:

```powershell
python -m iot.local_replay --limit 5
```

Use a custom training file:

```powershell
python -m iot.local_replay --train-path data/train_FD001.txt --limit 10
```

Run the IoT unit tests:

```powershell
python -m unittest discover -s tests
```

Run the milestone verification checks:

```powershell
python scripts/verify_preprocessing.py
python scripts/verify_lstm_baseline.py
```

## MQTT Sensor Publisher

Milestone 4 publishes the same validated FD001 training replay messages to MQTT.
The publisher reuses `iot.local_replay` and `iot.sensor_message`, so published
payloads contain only `unit_id`, `cycle`, and the 17 selected sensor features.
It does not publish RUL targets, labels, future cycles, official test targets,
or InfluxDB data.

Install the MQTT client dependency before real broker publishing:

```powershell
python -m pip install paho-mqtt
```

Check whether Mosquitto is already available on Windows:

```powershell
Get-Command mosquitto
Get-Command mosquitto_sub
```

If those commands are not found, install Mosquitto with the Windows installer
from the official Eclipse Mosquitto download page, then open a new PowerShell
window so the updated `PATH` is visible. Start a local broker in one terminal:

```powershell
mosquitto -v
```

In a second terminal, subscribe to the test topic:

```powershell
mosquitto_sub -h localhost -p 1883 -t industrial/test -C 1
```

Run a dry-run without Paho or a broker:

```powershell
python -m iot.mqtt_publisher --dry-run --limit 5
```

Publish a small replay to a local broker:

```powershell
python -m iot.mqtt_publisher --host localhost --port 1883 --topic industrial/test --qos 0 --limit 5
```

Verify end-to-end broker delivery with an in-process MQTT subscriber:

```powershell
python -m iot.mqtt_publisher --host localhost --port 1883 --topic industrial/test --qos 0 --limit 1 --verify-delivery
```

Successful verification prints the topic and the approved payload fields:
`unit_id`, `cycle`, and the 17 selected sensor fields. If the broker is down,
the host or port is wrong, or the message is not received before the timeout,
the command exits non-zero with an `MQTT publisher error` message.

You can also verify manually by comparing the subscriber output to the approved
field list. A valid payload must not contain `rul`, `failure_label`,
`timestamp`, InfluxDB fields, or any unselected FD001 sensors.

Use a shorter timeout when checking connection failures:

```powershell
python -m iot.mqtt_publisher --host localhost --port 1883 --topic industrial/test --limit 1 --verify-delivery --verify-timeout 2
```

Configuration can also come from environment variables:

```powershell
$env:IOT_MQTT_HOST = "localhost"
$env:IOT_MQTT_PORT = "1883"
$env:IOT_MQTT_TOPIC = "industrial/test"
$env:IOT_MQTT_QOS = "0"
$env:IOT_REPLAY_LIMIT = "5"
python -m iot.mqtt_publisher
```

Optional environment variables:

- `IOT_TRAIN_PATH`: defaults to `data/train_FD001.txt`.
- `IOT_REPLAY_LIMIT`: defaults to replaying the full training file.
