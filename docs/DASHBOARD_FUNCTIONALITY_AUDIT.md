# Dashboard Functionality Audit

Audit date: 2026-11-08
Repository: `C:\Users\moham\Desktop\IOT_Integrated`

## Scope and limitation

Audited the complete current dashboard surface in `app/static/index.html`, `app/static/app.js`, `app/static/styles.css`, and the FastAPI routes in `app/main.py`. Ran backend endpoint tests, API error/empty-state checks, JavaScript syntax validation, and backend unit tests.

The Windows browser-automation surface was unavailable during this audit: no browser was exposed by the Computer Use runtime. Therefore, browser-rendered screenshots, DevTools console output, keyboard/focus behavior, responsive layout in an actual browser, and network-panel inspection could not be automated. Those items are explicitly listed as manual verification steps below. No browser-console errors are claimed either way.

## Dashboard inventory

The UI is a single static page served at `/` with these visible sections:

1. Header and backend health pill.
2. Four readiness cards: Sensor, NLP, Fusion, IoT.
3. Technician Log Analysis form and structured result panel.
4. Sensor Inference form and result panel.
5. Fusion Readiness panel.
6. Model Modes panel.
7. Footer describing local development mode.

There are no separate UI routes/pages, charts, prediction-history tables, machine selector, settings form, documentation navigation, authentication controls, or live IoT controls in the current dashboard.

## Feature-by-feature results

| Feature/control | UI behavior | Backend endpoint | Result |
|---|---|---|---|
| Backend health pill | Calls health during page load and shows healthy/unavailable | `GET /api/health` | Working; actual response-driven |
| Readiness cards | Loads sensor/NLP/fusion/IoT state | `GET /api/status` | Working; actual response-driven |
| Model modes list | Renders model name/status rows | `GET /api/models` | Working; actual response-driven |
| Technician log ID | Input value is sent as `log_id` | `POST /api/technician-log/analyze` | Working |
| Technician text | Required client-side; sent as `text` | `POST /api/technician-log/analyze` | Working |
| Machine/entity ID | Optional value sent as `unit_id` | `POST /api/technician-log/analyze` | Transported; backend currently performs no machine lookup |
| Analyze button | Calls endpoint and prints JSON result | `POST /api/technician-log/analyze` | Working |
| Empty technician text | Prevents request and shows an inline message | None | Working empty state |
| Technician result | Shows normalized record, entities, severity, embedding diagnostics | Response body | Working; no fabricated embedding |
| Sensor JSON input | Newly added textarea for a record array | — | Working input surface |
| Run sensor inference | Parses JSON and calls backend | `POST /api/sensor/infer` | Working control and endpoint integration |
| Empty sensor input | Prevents request and explains required input | None | Working empty state |
| Invalid sensor JSON | Prevents request and displays parse error | None | Working client error state |
| Sensor result | Prints prediction or backend error JSON | Response body | Working; requires valid 30-cycle records/checkpoint |
| Fusion readiness | Displays explicit unavailable/not-validated status | `GET /api/status`; `POST /api/fusion/infer` exists but is not called by the UI | Honest but incomplete: no fusion submit form because no validated model exists |
| Fusion reason | Uses backend status reason | `GET /api/status` | Working; actual response-driven |
| IoT status | Shows local replay availability and “services not checked” | `GET /api/status` | Honest; no live-service claim |
| Footer | Static explanatory text | None | Informational |

## Backend endpoint results

Automated `TestClient` checks:

```text
GET  /api/health                         200
GET  /api/status                         200
GET  /api/models                         200
POST /api/technician-log/analyze (empty) 200
POST /api/technician-log/analyze (valid) 200
POST /api/sensor/infer (empty records)   422
POST /api/fusion/infer                   200
```

The API test module passed:

```text
3 passed in 3.18s
```

Full backend/project suite remains:

```text
368 passed, 3 skipped, 1 warning, 25 subtests passed
```

The existing warning is `tests/test_influx_ingestion.py::test_config` returning a configuration object instead of `None`; it is unrelated to the dashboard.

### Endpoint/data integrity observations

- `/api/status` derives sensor checkpoint/scaler readiness from filesystem presence, reports NLP embedding diagnostics, and explicitly reports fusion as `not_validated`.
- `/api/models` reports the configured model modes and current availability.
- `/api/technician-log/analyze` calls the existing preprocessing, rule-NER, and severity modules. Its embedding block correctly reports `not_generated` when the optional local transformer stack is unavailable.
- `/api/sensor/infer` calls the existing `iot.lstm_inference.predict_records`; it does not generate sensor values and rejects invalid windows.
- `/api/fusion/infer` intentionally returns `prediction: null` and `status: not_validated`; it does not fabricate a fused result.
- No API response exposes MQTT tokens, Influx tokens, passwords, or other secrets.

## Defect found and fixed

### Missing sensor form/action

Before this audit, the Sensor Inference panel stated that users could submit a 30-cycle window but contained no textarea, submit button, or API call. This was a genuine UI/backend integration defect.

Fixed minimally:

- Added `#sensorRecords` JSON textarea.
- Added `#inferSensor` button.
- Added JSON parsing and validation errors in `app/static/app.js`.
- Added `POST /api/sensor/infer` invocation and response rendering.
- Preserved the existing backend contract and model behavior.
- Added empty-input and invalid-JSON states.
- Added API coverage in `tests/test_api.py` for the sensor error path and existing API behavior.

### Health-load robustness

The page previously assumed status loading implied health. It now calls `/api/health` first and displays a backend-unavailable state if health or status loading fails. It still does not hide the fact that model/fusion services may independently be unavailable.

## Placeholders and nonfunctional/incomplete controls

- The machine/entity input is accepted and returned as record data but is not connected to a machine registry or lookup endpoint.
- Sensor inference requires users to paste a correctly shaped JSON array; there is no UI replay browser, sample loader, or machine selector.
- Fusion has no submit control. This is intentional because no approved trained fusion checkpoint exists, but the panel is readiness-only.
- There are no charts. No chart is misleadingly shown, which is correct; history/trend data APIs do not exist.
- There is no prediction history despite the broader application goal. No history endpoint or local persistence layer is currently exposed.
- IoT cards do not perform live MQTT/Influx health checks; the UI correctly says `not_checked`.
- The model list is status metadata only; it does not expose training runs, checkpoint hashes, or metric drill-down.
- The footer and explanatory copy are static, not linked to the documentation pages.
- No frontend build tool is present; the UI is plain static HTML/CSS/JavaScript. `node --check app/static/app.js` passed.

## Loading, empty, success, and error states

### Automated coverage

- Loading: page initialization performs health, status, and model requests; a browser-rendered spinner is not implemented.
- Backend unavailable: `load()` catches failures, changes the health pill to `Backend unavailable`, and renders an `ERROR` card.
- Empty technician text: client-side message, no request.
- Valid technician text: API response includes normalized text, rule entities, severity, and embedding diagnostics.
- Empty sensor input: client-side message, no request.
- Invalid sensor JSON: client-side parse error, no request.
- Invalid sensor payload: API returns 422 and UI prints returned JSON.
- Missing sensor artifacts: API returns 503 and UI prints returned JSON.
- Fusion unavailable: API returns an explicit null prediction/not-validated body; UI status is readiness-only.

### Not automated

The actual DOM rendering, CSS overflow, focus behavior, responsive breakpoints, button click behavior in a real browser, browser console, and network panel could not be observed because no browser was available to the computer-use runtime.

## Browser console/backend error audit

Backend errors were inspected through `TestClient` and endpoint smoke requests. No unhandled backend exception was observed for the tested paths. The static JavaScript passed Node syntax checking.

Browser console inspection was blocked by the unavailable browser surface. No claim is made that the console is error-free until manual verification is performed.

## Remaining issues

1. Add a real browser-based smoke test using Playwright or a connected browser environment.
2. Add a visible refresh/status retry control if operators need recovery without reloading.
3. Add machine selection only after an authoritative machine/entity registry exists.
4. Add history/trend endpoints and charts only when actual stored prediction/sensor data is available.
5. Add a fusion form only after an approved trained fusion checkpoint and data contract exist.
6. Add real transformer embedding status/model controls only after pinned local model assets are supplied.
7. Resolve the unrelated pytest return-value warning.
8. Add API authentication and production deployment controls before binding beyond localhost.

## Exact manual verification steps

From PowerShell:

```powershell
cd C:\Users\moham\Desktop\IOT_Integrated
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser and manually verify:

1. DevTools Console is free of JavaScript errors on initial load.
2. Network panel shows successful `GET /api/health`, `/api/status`, and `/api/models`.
3. The four readiness cards match the JSON returned by `/api/status`.
4. Submit empty technician text and confirm no request is sent and the inline message appears.
5. Submit `Critical vibration near bearing` and confirm the result shows `critical`, `vibration`, and `bearing` from the API response.
6. Paste invalid sensor JSON and confirm the client-side error appears.
7. Paste a valid 30-cycle, 17-feature sensor-record array and confirm `POST /api/sensor/infer` returns a backend prediction; do not invent records—use the project replay output.
8. Temporarily stop the backend, reload, and confirm the backend-unavailable card appears; restart it and reload.
9. Confirm the fusion panel says `NOT VALIDATED` and never displays a numeric fused prediction.
10. Resize the browser to desktop/tablet/mobile widths and check that cards, forms, and result blocks do not overflow.
11. Confirm the UI does not display any MQTT/Influx token or secret.

## Files changed by this audit

- `app/static/index.html`: added sensor JSON input and inference button.
- `app/static/app.js`: added health-first loading, sensor request handling, empty/error states.
- `docs/DASHBOARD_FUNCTIONALITY_AUDIT.md`: this report.

No models, datasets, frozen artifacts, checkpoints, project contracts, or Git history were modified. No commit or push was performed.
