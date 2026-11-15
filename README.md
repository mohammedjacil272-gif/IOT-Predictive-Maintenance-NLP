# IoT Predictive Maintenance: Current Status

## Portable MaintainIQ demonstration

The Windows-first desktop demo provides a focused workflow: select a registered machine, run the machine-specific analyst, inspect an independent FD001 LSTM RUL result and rule-based technician findings, and optionally receive an evidence-grounded explanation from local Qwen through Ollama. Start with [the portable demo setup guide](docs/DEMO_SETUP_GUIDE.md) and [the evaluator script](docs/DEMO_SCRIPT.md).

The reproducible demo uses existing FD001 benchmark rows under an explicitly synthetic machine identity. It is not live industrial telemetry and does not claim trained multimodal fusion. The LSTM owns the numerical RUL; NLP owns technician-log extraction; Qwen explains only the supplied structured evidence.

The project now includes a tested local technician-assistant backend foundation: explicit sensor/NLP/fusion modes, local history, assistant analysis and confirmation contracts, typed transcription fallback, machine-health evidence separation, attachment validation, and transactional SQLite storage. Genuine multimodal fusion remains gated and unavailable.

A Windows-first React/Tauri desktop application is present under `frontend/` and `src-tauri/`. A native Windows executable and NSIS installer were built on 2026-11-10; clean-machine installation and visual native-UI inspection remain unverified.

This repository contains an FD001 sensor preprocessing and LSTM baseline pipeline, technician-log NLP and embedding artifact generation, metadata-based multimodal fusion components, and IoT ingestion/replay code. Software tests exercise selected interfaces; they do not establish real sensor/log pairing, live-service operation, or predictive improvement.

## Project status

| Area | Status in the reviewed repository |
|---|---|
| Sensor baseline | FD001 preprocessing, LSTM, and sensor embedding components exist. The frozen `best_model.pt`, `last_model.pt`, window arrays, and sensor embeddings are present locally and match the handover hashes. Do not retrain or generate replacements. |
| NLP | Log preprocessing, rule-based entities/severity, and embedding artifact generation exist. The five-log fixture is synthetic. |
| Fusion | Sensor and text records are aligned by metadata using the provisional `(unit_id, cycle)` key. No row-order or vector-position join is valid. Entity records must carry their own matching `log_id`. |
| Genuine data | No approved genuine paired dataset, authoritative cross-source equipment mapping, or genuine training authorization is established. Genuine training/evaluation remain blocked by the gates in the approval checklist and decision register. |
| Policies | Missing-log, multiple-log, and `has_log` choices are explicit implementation options, not production approvals. |
| IoT | MQTT, InfluxDB, and replay code and software/mock tests exist. No live service, device, or hardware validation is established. |
| Evaluation | No genuine fusion performance result or official FD001 test evaluation is established. Do not use official test labels for training or model selection. |

See the [approval checklist](context/MULTIMODAL_APPROVAL_CHECKLIST.md), [decision register](docs/MULTIMODAL_DECISION_REGISTER.md), [fusion contract](context/MULTIMODAL_FUSION_DATA_CONTRACT.md), [evaluation protocol](docs/MULTIMODAL_EVALUATION_PROTOCOL.md), and [next-phase handover](docs/FINAL_HANDOVER_NEXT_10_PHASES.md) for detailed current gates. Test reports are evidence tied to their recorded revision, environment, and command.

## Integration verification

From the repository root, use the project Python environment and run:

```powershell
python -m pytest -q
python scripts/verify_sensor_embeddings.py
python scripts/verify_fusion.py
python scripts/verify_fusion_provenance.py
python scripts/verify_nlp_fd001_compatibility.py
python scripts/verify_preprocessing.py
python scripts/verify_lstm_baseline.py
# After an owner supplies approved genuine logs:
python scripts/validate_genuine_logs.py path/to/approved_logs.csv --dataset owner-export --dataset-version 2026-01
```

The sensor verifier is read-only and checks the 64-D array, 17,731-row
metadata/index contract, `(unit_id, label_cycle)` uniqueness, and train/validation
alignment. Missing logs use an explicitly selected policy; the approved production
choice is still pending. The checked-in NLP package is a synthetic fixture, so
these commands validate interfaces and integrity only. They do not establish
genuine sensor/log pairing or multimodal performance. See
`context/WAHID_INTEGRATION_EXECUTION_REPORT.md` and
`docs/NLP_GENUINE_DATA_INTAKE.md` for the latest execution record and genuine
intake contract.

## Implemented flow

```text
FD001 sensor rows -> preprocessing/windows -> LSTM -> 64-D sensor embeddings
Technician logs -> preprocessing/entities/severity -> 768-D text embeddings
Sensor + log metadata -> provenance/policy checks -> (unit_id, cycle) fusion
```

The alignment key and real-world cycle relationship remain provisional until verified by data owners. The fixture can validate parsing, shapes, alignment, provenance behavior, and missing-text paths; it cannot establish real pairing, point-in-time availability, validated severity labels, or model improvement. Genuine records require additional source, artifact, timestamp, mapping, and approval evidence.

The EDP wind-farm dataset is a separately audited candidate. Raw workbooks were
not obtained in the current environment, so no EDP schema or fusion result is
claimed. See [the EDP audit](docs/EDP_DATASET_AUDIT.md), [the dataset contract](docs/MULTIMODAL_DATA_CONTRACT.md), and [the final multimodal handover](docs/FINAL_MULTIMODAL_HANDOVER.md).

## Data and safe boundaries

The tracked FD001 directory in this source snapshot is lowercase `data/`. Historical notes that say `Data/` are not current path instructions. Do not regenerate frozen baseline data, transforms, metrics, or checkpoints during documentation work. Check output paths before running commands that write artifacts.

`requirements.txt` declares the current broad runtime ranges; it is not a fully resolved lockfile. The verified handover environment is documented in `docs/WAHID_DEPENDENCY_SETUP.md`. The text model's immutable revision and asset-checksum lineage remain unresolved for the synthetic fixture.

For recorded test scope and results, see [readiness validation](docs/READINESS_VALIDATION_RESULTS.md) and the [next-phase handover](docs/FINAL_HANDOVER_NEXT_10_PHASES.md).
