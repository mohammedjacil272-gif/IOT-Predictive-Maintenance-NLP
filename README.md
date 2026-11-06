# IoT Predictive Maintenance: Current Status

This repository contains an FD001 sensor preprocessing and LSTM baseline pipeline, technician-log NLP and embedding artifact generation, metadata-based multimodal fusion components, and IoT ingestion/replay code. Software tests exercise selected interfaces; they do not establish real sensor/log pairing, live-service operation, or predictive improvement.

## Project status

| Area | Status in the reviewed repository |
|---|---|
| Sensor baseline | FD001 preprocessing, LSTM, and sensor embedding components exist. The expected `best_model.pt` checkpoint is absent from the checked-in inventory. Do not retrain or generate a replacement to clear this gap. |
| NLP | Log preprocessing, rule-based entities/severity, and embedding artifact generation exist. The five-log fixture is synthetic. |
| Fusion | Sensor and text records are aligned by metadata using the provisional `(unit_id, cycle)` key. No row-order or vector-position join is valid. Entity records must carry their own matching `log_id`. |
| Genuine data | No approved genuine paired dataset, authoritative cross-source equipment mapping, or genuine training authorization is established. Genuine training/evaluation remain blocked by the gates in the approval checklist and decision register. |
| Policies | Missing-log, multiple-log, and `has_log` choices are explicit implementation options, not production approvals. |
| IoT | MQTT, InfluxDB, and replay code and software/mock tests exist. No live service, device, or hardware validation is established. |
| Evaluation | No genuine fusion performance result or official FD001 test evaluation is established. Do not use official test labels for training or model selection. |

See the [approval checklist](context/MULTIMODAL_APPROVAL_CHECKLIST.md), [decision register](docs/MULTIMODAL_DECISION_REGISTER.md), [fusion contract](context/MULTIMODAL_FUSION_DATA_CONTRACT.md), [evaluation protocol](docs/MULTIMODAL_EVALUATION_PROTOCOL.md), and [fusion handoff](docs/MULTIMODAL_FUSION_HANDOFF.md) for detailed current gates. Test reports are historical evidence tied to their recorded revision, environment, and command.

## Implemented flow

```text
FD001 sensor rows -> preprocessing/windows -> LSTM -> 64-D sensor embeddings
Technician logs -> preprocessing/entities/severity -> 768-D text embeddings
Sensor + log metadata -> provenance/policy checks -> (unit_id, cycle) fusion
```

The alignment key and real-world cycle relationship remain provisional until verified by data owners. The fixture can validate parsing, shapes, alignment, provenance behavior, and missing-text paths; it cannot establish real pairing, point-in-time availability, validated severity labels, or model improvement. Genuine records require additional source, artifact, timestamp, mapping, and approval evidence.

## Data and safe boundaries

The tracked FD001 directory in this source snapshot is lowercase `data/`. Historical notes that say `Data/` are not current path instructions. Do not regenerate frozen baseline data, transforms, metrics, or checkpoints during documentation work. Check output paths before running commands that write artifacts.

No approved dependency manifest or platform lock exists in this base snapshot. Versions reported in audit documents are observations, not supported installation pins. Defer an installable `requirements.txt` until maintainers approve a runtime/platform and validate dependencies in a clean environment. The text model's immutable revision and asset-checksum lineage also remain unresolved for existing artifacts.

For recorded test scope and results, see [readiness validation](docs/READINESS_VALIDATION_RESULTS.md). No tests were run to prepare this proposal.
