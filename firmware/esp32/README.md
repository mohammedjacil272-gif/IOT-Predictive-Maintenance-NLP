# ESP32 firmware project

This Arduino-compatible ESP32 firmware publishes synthetic MaintainIQ hardware
telemetry. It does not require physical sensors and does not produce FD001 data.

1. Install PlatformIO Core or the VS Code PlatformIO extension.
2. Copy `config.example.h` to `config.h`; set Wi-Fi and broker values. `config.h`
   is ignored by Git.
3. Build/upload with `pio run -t upload`, then open `pio device monitor`.

Libraries are declared in `platformio.ini`: Espressif32 Arduino core,
PubSubClient, and ArduinoJson. Arduino IDE users must install the equivalent
libraries and select an ESP32 Dev Module board. The code expects a synchronized
UTC clock and waits for NTP before publishing. The broker topic defaults to
`maintainiq/hardware/telemetry` and the interval is 5 seconds.

`diagram.json` and `wokwi.toml` provide an ESP32 DevKit simulation target. A
Wokwi run requires a firmware binary/ELF build and valid simulator network
configuration. External broker access depends on the Wokwi networking plan and
gateway setup; this repository has not verified it.
