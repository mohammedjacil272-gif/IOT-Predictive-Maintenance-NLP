# IoT live connection setup

The MQTT publisher and InfluxDB ingestion paths are implemented and covered by mock/unit tests. Live verification remains blocked because this environment has no authorized service variables set.

Required non-production configuration, supplied by Jacil or the service owner through a protected environment only:

```text
IOT_MQTT_HOST, IOT_MQTT_PORT, IOT_MQTT_TOPIC, IOT_MQTT_QOS
IOT_INFLUXDB_URL, IOT_INFLUXDB_ORG, IOT_INFLUXDB_BUCKET, IOT_INFLUXDB_TOKEN
IOT_INFLUXDB_MEASUREMENT
```

Before a live test, confirm an isolated broker topic and Influx bucket, credentials, retention/cleanup procedure, TLS/authentication requirements, and permission to publish/query. Never use production topics or buckets. Tokens must not appear in logs or reports. Run the existing IoT tests first:

```powershell
python -m pytest tests/test_sensor_message.py tests/test_local_replay.py tests/test_mqtt_publisher.py tests/test_influx_ingestion.py -q
```

Then record publish acknowledgement, ingestion result, query result, service failures, and cleanup evidence separately from mock-test results. No live publish or database write was performed in this handover.
