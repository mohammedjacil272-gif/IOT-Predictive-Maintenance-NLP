#include <Arduino.h>
#include <math.h>
#include <time.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "config.h"

static const char* SCHEMA_VERSION = "1.0";
static const uint32_t KEEPALIVE_SECONDS = 30;
static const uint32_t WIFI_RETRY_MS = 10000;
static const uint32_t MQTT_RETRY_MS = 5000;

WiFiClient networkClient;
PubSubClient mqtt(networkClient);
String sessionId;
uint32_t sequenceNumber = 0;
uint32_t lastPublishMs = 0;
uint32_t lastWifiAttemptMs = 0;
uint32_t lastMqttAttemptMs = 0;
uint32_t simulationStartMs = 0;
bool wifiAttempted = false;
bool mqttAttempted = false;

struct Reading { float temperature; float vibration; int rpm; float pressure; const char* status; };

// Deterministic 19-sample demonstration cycle: normal, ramp, abnormal, failure, recovery.
Reading syntheticReading(uint32_t sample) {
  const uint32_t p = sample % 19;
  const char* status = "normal";
  float severity = 0.0f;
  if (p < 6) { status = "normal"; severity = 0.0f; }
  else if (p < 10) { status = "warming"; severity = 0.15f + 0.35f * ((p - 6) / 3.0f); }
  else if (p < 14) { status = "abnormal"; severity = 0.55f + 0.35f * ((p - 10) / 3.0f); }
  else if (p < 17) { status = "failure"; severity = 1.0f; }
  else { status = "recovery"; severity = 0.65f * (1.0f - ((p - 17) / 1.0f)); }
  // Small deterministic ripple keeps the trace from looking perfectly flat.
  float ripple = (p % 2 == 0) ? 0.04f : -0.04f;
  Reading r;
  r.temperature = 45.0f + 50.0f * severity + ripple;
  r.vibration = 1.2f + 7.3f * severity + ripple;
  r.rpm = (int)roundf(1500.0f + 600.0f * severity);
  r.pressure = 3.2f + 2.6f * severity + ripple * 0.25f;
  r.status = status;
  return r;
}

String makeSessionId() {
  uint64_t chip = ESP.getEfuseMac();
  char value[40];
  snprintf(value, sizeof(value), "%04X%08X-%08lX", (uint16_t)(chip >> 32), (uint32_t)chip,
           (unsigned long)esp_random());
  return String(value);
}

String utcTimestamp() {
  time_t now = time(nullptr);
  if (now < 1700000000) return String("");  // Wait for NTP; never publish a fabricated event time.
  struct tm timeinfo;
  gmtime_r(&now, &timeinfo);
  char value[25];
  strftime(value, sizeof(value), "%Y-%m-%dT%H:%M:%SZ", &timeinfo);
  return String(value);
}

void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  if (wifiAttempted && millis() - lastWifiAttemptMs < WIFI_RETRY_MS) return;
  wifiAttempted = true;
  lastWifiAttemptMs = millis();
  Serial.printf("Connecting to Wi-Fi SSID %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void ensureMqtt() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
  if (mqttAttempted && millis() - lastMqttAttemptMs < MQTT_RETRY_MS) return;
  mqttAttempted = true;
  lastMqttAttemptMs = millis();
  Serial.printf("Connecting to MQTT %s:%d...\n", MQTT_HOST, MQTT_PORT);
  bool connected = MQTT_USERNAME[0]
      ? mqtt.connect(DEVICE_ID, MQTT_USERNAME, MQTT_PASSWORD)
      : mqtt.connect(DEVICE_ID);
  Serial.println(connected ? "MQTT connected" : "MQTT connection failed");
}

void setup() {
  Serial.begin(115200);
  delay(200);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setKeepAlive(KEEPALIVE_SECONDS);
  sessionId = makeSessionId();
  simulationStartMs = millis();
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  Serial.printf("MaintainIQ device %s session %s\n", DEVICE_ID, sessionId.c_str());
  ensureWifi();
}

void loop() {
  ensureWifi();
  ensureMqtt();
  if (mqtt.connected()) mqtt.loop();
  if (millis() - lastPublishMs < PUBLISH_INTERVAL_MS) return;
  lastPublishMs = millis();
  if (!mqtt.connected()) return;

  Reading r = syntheticReading((millis() - simulationStartMs) / PUBLISH_INTERVAL_MS);
  String timestamp = utcTimestamp();
  if (timestamp.isEmpty()) {
    Serial.println("Waiting for synchronized UTC time before publishing");
    return;
  }
  StaticJsonDocument<512> doc;
  doc["schema_version"] = SCHEMA_VERSION;
  doc["device_id"] = DEVICE_ID;
  doc["session_id"] = sessionId;
  doc["timestamp"] = timestamp;
  doc["sequence_number"] = ++sequenceNumber;
  doc["temperature"] = r.temperature;
  doc["vibration"] = r.vibration;
  doc["rpm"] = r.rpm;
  doc["pressure"] = r.pressure;
  doc["operating_status"] = r.status;
  char payload[512];
  size_t length = serializeJson(doc, payload, sizeof(payload));
  if (length == 0 || !mqtt.publish(MQTT_TOPIC, payload, false)) {
    Serial.println("Telemetry publish failed; sequence will advance on next reading");
    return;
  }
  Serial.printf("Published %s seq=%lu status=%s temp=%.2f vibration=%.2f rpm=%d pressure=%.2f\n",
                MQTT_TOPIC, (unsigned long)sequenceNumber, r.status, r.temperature,
                r.vibration, r.rpm, r.pressure);
}
