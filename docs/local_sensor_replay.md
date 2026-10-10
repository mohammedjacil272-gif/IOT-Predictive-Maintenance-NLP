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
