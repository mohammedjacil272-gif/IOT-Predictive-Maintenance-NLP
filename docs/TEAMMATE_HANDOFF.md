# MaintainIQ ESP32 module handoff

## Included

This feature adds ESP32 Arduino firmware that creates a deterministic synthetic
normal-to-failure-to-recovery trace, Wi-Fi and MQTT publishing, a strict
versioned Python message contract, an MQTT-to-Influx hardware service, mockable
storage boundaries, tests, and Wokwi project descriptors. The hardware path is
independent of FD001.

## Teammate configuration and entry points

1. Choose the test broker, hardware Influx URL/org/bucket, credentials, token
   scope, retention, and deployment TLS/network policy.
2. Configure the firmware in `firmware/esp32/config.h` (copy from the example;
   never commit local credentials).
3. Configure the `IOT_HARDWARE_MQTT_*` and `IOT_HARDWARE_INFLUXDB_*` environment
   variables and install approved `paho-mqtt` and `influxdb-client` runtime
   dependencies.
4. Start the backend with `python -m iot.hardware.mqtt_ingestion`.
5. Upload firmware following `docs/ESP32_SETUP.md`; inspect the serial output
   for the topic and message sequence.
6. Run unit tests with `python -m pytest tests/test_hardware_*.py -q`.

Contract details are in `docs/HARDWARE_TELEMETRY_SCHEMA.md`; full setup and
service configuration are in `docs/ESP32_SETUP.md` and
`docs/BACKEND_INTEGRATION_GUIDE.md`.

## Verification state and limits

Python unit tests use local fakes/mocks and do not contact a broker or database.
Firmware build, Wokwi execution, external MQTT routing, and physical ESP32
behavior must be verified in the teammate's approved environment. No physical
device test is claimed here. Synthetic demonstration thresholds are not
engineering limits. The starter firmware does not implement TLS. Deduplication
is process-local. PubSubClient sends QoS 0 and does not queue samples during a
broker outage; failed publications are logged and later samples are attempted
after reconnection. The backend uses Paho's default message acknowledgement;
an Influx write failure is logged and is not guaranteed to cause broker
redelivery. Reliable delivery needs a separately configured persistent session,
manual-ack policy, or device outbox.

The existing `fd001_sensor_reading` measurement, 17-feature ordering,
preprocessing, 30-cycle inference window, and frozen artifacts are unchanged.
No direct mapping to the frozen LSTM is provided or implied.
