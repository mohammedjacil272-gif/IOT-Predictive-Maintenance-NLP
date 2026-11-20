# Multimodal dataset selection

Status: **no final genuine training dataset selected**.

| Candidate | Verified role | Decision |
|---|---|---|
| EDP Portugal wind farm | Version 1 and a later version 2 record describe SCADA, event remarks, and confirmed failure logs under CC BY 4.0. Raw files were not accessible to this run for schema validation. | Primary candidate pending explicit version selection and raw audit |
| Swedish paper-machine dataset | Genuine vibration plus Swedish expert technical annotations; language and published association policy require review. | Secondary research candidate |
| Azure PdM | Structured telemetry, errors, maintenance, failures, machine metadata; not rich technician text. | Sensor/event benchmark only |
| Factory Sensors with Failure Labels | Publicly described synthetic multi-table data. | Synthetic integration fixture only |
| Industrial Maintenance Synthetic | Generated sensor tags and work-order text. | Synthetic NLP/ETL robustness only |
| UTK-ASL | Sensor dataset candidate; no verified technician-text pairing established. | Sensor-only candidate |

EDP is not forced into the pipeline until raw workbooks, license attribution,
timestamps, text quality, and leakage-safe relationships are independently
verified. Unrelated datasets will not be joined by name or fabricated IDs.
