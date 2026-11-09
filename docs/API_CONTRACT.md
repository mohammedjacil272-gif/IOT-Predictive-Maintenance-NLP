# Local API contract

Start with `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.

- `GET /api/health`: local service health.
- `GET /api/status`: sensor/NLP/fusion/IoT readiness without secrets.
- `GET /api/models`: model modes and availability.
- `POST /api/sensor/infer`: `{ "records": [...] }`; requires a valid 30-cycle FD001 window and local checkpoint/scaler.
- `POST /api/technician-log/analyze`: log ID, text, optional unit and timestamps; returns normalized record, rule entities, severity, and embedding diagnostics.
- `POST /api/fusion/infer`: intentionally returns `not_validated` until an approved trained fusion artifact exists.
- `GET /api/diagnostics`: local paths/runtime diagnostics; no credentials.

All inference errors are returned as structured HTTP errors. The API has database-disabled local mode and does not contact MQTT/Influx merely to launch.
# API contract

`GET /api/status` and `GET /api/fusion/readiness` expose readiness without fabricating results. Official FD001 test evaluation is `not_run` unless an explicit `artifacts/metrics/fd001_lstm_baseline/test_metrics.json` artifact is present; validation metrics are reported separately and never imply official-test completion. `POST /api/sensor/infer` is `sensor_only`; `POST /api/technician-log/analyze` is `nlp_only`; `POST /api/fusion/infer` remains unavailable until a verified paired checkpoint exists and returns `prediction: null`. MQTT and InfluxDB are not claimed as live services.

`GET /api/attachments/contract` is the shared attachment contract. Uploads accept
JPEG, PNG, PDF, TXT, CSV, and JSON with a 10 MiB maximum. Text uploads must be
valid UTF-8; CSV and JSON are parsed before storage. The filename extension and
declared MIME type must agree. Files are stored under local application data,
outside static assets, and are marked `stored_unanalyzed`.
