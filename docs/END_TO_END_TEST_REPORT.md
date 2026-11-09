# End-to-end test report

API-level health, static dashboard response, sensor validation, technician analysis, fusion readiness, unified dispatch, local history persistence/clear, malformed input, missing dependency and invalid mode tests are covered. Browser automation was not installed or used; manual browser validation remains. Full suite and project verifiers are recorded in the final demonstration report.
## Current verification refresh — 2026-11-09

The current worktree was revalidated after the earlier report. The full Python
suite passed with **429 passed, 4 skipped, 2 warnings, 25 subtests**. Targeted
API, assistant, sensor-message, Influx, and MQTT regression tests passed with
**50 passed, 1 warning**. TypeScript and the frontend production build passed.

The API now reports official FD001 test evaluation as `not_run` unless an
explicit test metrics artifact exists; validation metrics are reported under a
separate validation split. The attachment contract accepts and validates
TXT/CSV/JSON as well as existing PDF/JPEG/PNG formats. Real IoT payloads may
provide an offset-aware `event_timestamp`; cycle-derived timestamps remain
explicitly synthetic for FD001 replay only.

Genuine paired sensor/log end-to-end fusion remains blocked because no approved
paired dataset or fusion checkpoint is available. No official test evaluation,
live broker/database connection, or Tauri build was run in this environment.
