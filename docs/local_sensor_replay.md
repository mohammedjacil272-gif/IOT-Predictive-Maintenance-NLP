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

Milestone 3 publishes the same validated FD001 training replay messages to MQTT.
The publisher reuses `iot.local_replay` and `iot.sensor_message`, so published
payloads contain only `unit_id`, `cycle`, and the 17 selected sensor features.
It does not publish RUL targets, labels, future cycles, official test targets,
or InfluxDB data.

Install the MQTT client dependency before real broker publishing:

```powershell
python -m pip install paho-mqtt
```

Run a dry-run without Paho or a broker:

```powershell
python -m iot.mqtt_publisher --dry-run --limit 5
```

Publish a small replay to a local broker:

```powershell
python -m iot.mqtt_publisher --host localhost --port 1883 --topic industrial/sensors --qos 0 --limit 5
```

Configuration can also come from environment variables:

```powershell
$env:IOT_MQTT_HOST = "localhost"
$env:IOT_MQTT_PORT = "1883"
$env:IOT_MQTT_TOPIC = "industrial/sensors"
$env:IOT_MQTT_QOS = "0"
$env:IOT_REPLAY_LIMIT = "5"
python -m iot.mqtt_publisher
```

Optional environment variables:

- `IOT_TRAIN_PATH`: defaults to `data/train_FD001.txt`.
- `IOT_REPLAY_LIMIT`: defaults to replaying the full training file.
