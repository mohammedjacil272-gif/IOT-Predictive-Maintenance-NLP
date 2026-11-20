# ESP32 setup

## Hardware and toolchain

The primary target is an ESP32 DevKit compatible board; synthetic data means no
sensor wiring is required. Install Arduino IDE 2.x and the Espressif ESP32 board
package, or PlatformIO Core/VS Code. PlatformIO dependency pins are in
`firmware/esp32/platformio.ini` (ESP32 Arduino framework, PubSubClient, and
ArduinoJson). Arduino IDE users need the equivalent two libraries.

## Configure and upload

1. Copy `firmware/esp32/config.example.h` to `firmware/esp32/config.h`.
2. Set `WIFI_SSID`, `WIFI_PASSWORD`, `MQTT_HOST`, `MQTT_PORT`, optional MQTT
   username/password, `DEVICE_ID`, and optionally the topic/interval. Use the
   broker hostname/IP reachable from the ESP32; `localhost` means the device
   itself and is usually wrong.
3. Build and upload from the firmware directory with `pio run -t upload`.
   Alternatively open `esp32_telemetry.ino` in Arduino IDE, select an ESP32
   Dev Module and the detected serial port, then upload.
4. Open Serial Monitor at 115200 baud. Confirm Wi-Fi, MQTT connection, and
   `Published maintainiq/hardware/telemetry` lines.

The firmware starts NTP synchronization and waits for valid UTC time before
publishing, so a network that blocks NTP will not produce telemetry. Sequence
numbers advance on each publish attempt, so a failed attempt can leave a gap;
`session_id` changes on boot. The current demonstration uses an unauthenticated MQTT connection when
username is blank; use broker auth/TLS for shared networks (TLS support is not
implemented in this starter firmware).

## Simulation

The Wokwi ESP32 diagram and `wokwi.toml` are included. Build firmware first.
Simulation execution and external MQTT access have not been verified. Wokwi
external networking may require its gateway/network feature; do not assume a
local computer broker is reachable without configuring that route.

See `HARDWARE_TELEMETRY_SCHEMA.md` for fields and `BACKEND_INTEGRATION_GUIDE.md`
for the Python service.
