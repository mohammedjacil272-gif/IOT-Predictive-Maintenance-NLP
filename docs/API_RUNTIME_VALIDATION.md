# API runtime validation

FastAPI `TestClient` checks passed for `/`, `/api/status`, `/api/models`, `/api/fusion/readiness`, technician analysis, invalid empty inputs, sensor empty input, and unavailable fusion inference. The API reports sensor-only and NLP-only modes separately and returns `prediction: null` for unavailable fusion. No live MQTT/InfluxDB claim is made.
