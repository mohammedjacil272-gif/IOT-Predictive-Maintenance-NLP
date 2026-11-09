# Real data inventory

No approved operational paired dataset was found.

| Path | Contents | Records/shape | Classification | Identity/time/labels |
|---|---|---:|---|---|
| `data/train_FD001.txt` | CMAPSS FD001 train benchmark | 20,631 rows; 100 engines | PUBLIC_DATASET | FD001 engine IDs and cycles; RUL-derived benchmark target |
| `data/test_FD001.txt` | CMAPSS FD001 test benchmark | 13,096 rows; 100 engines | PUBLIC_DATASET | engine IDs/cycles; official test labels are separate and must not train/validate |
| `data/RUL_FD001.txt` | FD001 test RUL file | 100 labels | PUBLIC_DATASET | official test labels only |
| `tests/fixtures/technician_logs.csv` | technician-log fixture | 5 rows | SYNTHETIC | unit/cycle fields; no timestamps |
| `artifacts/data/fd001_nlp_fixture/*` | processed synthetic NLP fixture | 5 logs, 16 entities | SYNTHETIC | fixture IDs; no complete provenance/timestamps |
| `experiments/azure_pdm/tests/fixtures/PdM_*.csv` | Azure PdM test fixtures | 1–4 rows/file | SYNTHETIC_FIXTURE / PUBLIC_SCHEMA_FIXTURE | machineID/datetime fields; no approved project pairing |

Existing sensor embeddings are `artifacts/embeddings/fd001_lstm_baseline/sensor_embeddings.npy` (`17,731 x 64`) with metadata in the adjacent CSV. Existing integration-evaluation files are historical artifacts and are not evidence of genuine multimodal pairing. No database export, MongoDB source, SCANIA data, or approved crosswalk was found.
