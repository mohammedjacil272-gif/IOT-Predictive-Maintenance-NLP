# Backend integration guide

The backend modules are separate from the existing FD001 replay path:

- Contract: `iot/hardware/message.py`
- Subscriber/service entry point: `python -m iot.hardware.mqtt_ingestion`
- Storage adapter: `iot/hardware/influx_storage.py`
- Default topic: `maintainiq/hardware/telemetry`
- Default measurement: `hardware_sensor_reading`

The service uses lazy imports for `paho-mqtt` and `influxdb-client`; install
those packages in the approved backend environment before starting it. Set:

```text
IOT_HARDWARE_MQTT_HOST
IOT_HARDWARE_MQTT_PORT       (default 1883)
IOT_HARDWARE_MQTT_TOPIC      (default maintainiq/hardware/telemetry)
IOT_HARDWARE_MQTT_QOS        (default 1)
IOT_HARDWARE_MQTT_USERNAME   (optional)
IOT_HARDWARE_MQTT_PASSWORD   (optional)
IOT_HARDWARE_INFLUXDB_URL
IOT_HARDWARE_INFLUXDB_ORG
IOT_HARDWARE_INFLUXDB_BUCKET
IOT_HARDWARE_INFLUXDB_TOKEN
IOT_HARDWARE_INFLUXDB_MEASUREMENT (default hardware_sensor_reading)
```

Provide the MQTT/Influx values through the deployment's secret/environment
configuration; never commit credentials. Start with
`python -m iot.hardware.mqtt_ingestion`. Subscribe using the device topic and
ensure the broker allows the ESP32 publisher and backend subscriber. The service validates
each JSON message, drops malformed and duplicate/out-of-order messages, logs
failures, and continues receiving. Influx writes are synchronous; failed writes
do not advance the in-process sequence watermark. The current subscriber uses
the Paho default acknowledgement behavior, so a database failure is not
guaranteed to trigger broker redelivery. Durable retry requires a persistent
MQTT session/manual acknowledgement policy or an application-level outbox, and
should be agreed with the integrating team. The watermark is process-local.
The ESP32 uses PubSubClient QoS 0 and has no offline queue; telemetry produced
during a broker outage is not replayed after reconnection.

Influx stores device and session as tags; sensor values, operating status,
sequence number, and schema version are fields. The event UTC timestamp is the
point time. InfluxDB bucket retention, token scope, organization, network/TLS,
and broker authorization are deployment choices and must be configured by the
integrating team.

Example accepted message:

```json
{"schema_version":"1.0","device_id":"ESP32_001","session_id":"boot-42","timestamp":"2026-11-20T10:30:00Z","sequence_number":1,"temperature":45.6,"vibration":1.3,"rpm":1500,"pressure":3.2,"operating_status":"normal"}
```

This hardware schema is not accepted by `iot.sensor_message`, is not stored in
`fd001_sensor_reading`, and is not a valid direct input to the frozen FD001
LSTM. Any future ML adapter requires a separate engineering/data compatibility
review; synthetic demonstration values must not be represented as C-MAPSS
features.
