# IoT Predictive Maintenance: NLP and Multimodal Fusion

This repository contains the sensor-only FD001 LSTM baseline implementation and its recorded artifacts, a technician-log NLP artifact pipeline, and a metadata-aligned fusion adapter. The original baseline checkpoints are missing.

The current checkout is **not ready for genuine multimodal training**: no approved paired sensor/log dataset or genuine equipment mapping is present, timestamp and target policies remain pending, and no training authorization is recorded. Synthetic checks below establish software interface behavior only.

## Implemented architecture

```text
FD001 raw sensor rows
  -> validated preprocessing and train-fitted StandardScaler
  -> 30-cycle windows, 17 configured features
  -> 2-layer LSTM baseline (64-D embedding)
  -> recorded train/validation metrics and per-window metadata

Synthetic or approved technician log CSV
  -> original/normalized text
  -> deterministic rule-based entities and severity heuristic
  -> distilbert-base-uncased (768-D, attention-mask mean pooling; model/tokenizer revision not pinned)
  -> log-keyed artifacts and metadata

Sensor (unit_id, cycle) + technician log (unit_id, cycle)
  -> namespace, split, provenance, temporal and policy checks
  -> 64-D sensor + 768-D text vectors
  -> configurable MLP RUL regression model
```

Fusion alignment is by `(unit_id, cycle)`, never by CSV row order or vector position. Every entity must explicitly carry its own `log_id`, matching its parent log. Multiple-log pooling and missing-log handling require explicit policy arguments; current `mean_sorted_log_id` and `zero_vector` options are provisional and are not production approvals.

See [architecture](context/ARCHITECTURE.md), [approval checklist](context/MULTIMODAL_APPROVAL_CHECKLIST.md), and [provenance schema](context/MULTIMODAL_PROVENANCE_SCHEMA.md) for the current contracts and external blockers.

## Repository status

| Area | Status |
|---|---|
| FD001 preprocessing and baseline source | Implemented; checked-in train/validation metrics and data artifacts exist |
| Frozen baseline checkpoint | **Missing** (`best_model.pt`, `last_model.pt`); do not regenerate to fill the gap |
| NLP processing and artifact build | Implemented; rule-based and heuristic components; fixture is synthetic |
| Fusion adapter/model | Implemented and synthetically tested; no genuine paired-data evidence |
| Genuine fusion training | **Blocked** pending approved data, mapping, semantics, artifacts, policies, checkpoint and explicit authorization |
| MQTT, InfluxDB, MongoDB, Azure, Ollama, spaCy integration | Not implemented in this checkout; no live service test has been run |
| Official FD001 test evaluation | Not performed; do not use official test labels for development/model selection |

The checked-in validation metrics report sensor-only validation MAE 20.8844, RMSE 31.3649, and MSE 983.7574. They are existing recorded artifacts, not a result reproduced in this checkout session. No multimodal performance result exists.

## Environment

The repository was exercised in Python 3.11.9 with the versions in [`requirements.txt`](requirements.txt). These pins capture the observed test environment; they are not a cross-platform support guarantee. Install them in an isolated environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The NLP artifact builder uses Hugging Face `from_pretrained`; if the model is not cached, an explicit artifact-build run may access the network and download model files. The tests replace only model inference with deterministic fixture vectors and do not download models.

## Data and artifacts

FD001 raw files are under lowercase `data/`. The source pipeline's current default paths use `Data/`; on case-sensitive systems pass explicit lowercase paths. Preprocessing and baseline training write generated files and can replace outputs under `artifacts/`; inspect paths and obtain authorization before running them. The frozen baseline artifacts must not be regenerated or overwritten.

The NLP fixture is `tests/fixtures/technician_logs.csv`. Its identifiers, equipment/cycle values, log text, severity, and split are synthetic. The artifact builder writes processed CSV, entity JSONL, severity CSV, a 768-D tensor, metadata, and a verification report. Do not classify fixture records as genuine or use them as real model evidence.

## Safe verification

Run the test suite without pytest's cache provider:

```powershell
python -m pytest tests -q -p no:cacheprovider
```

Read-only fixture checks:

```powershell
python scripts/verify_nlp.py
python scripts/verify_fusion_provenance.py
```

The artifact builder, preprocessing command, baseline training/evaluation, multimodal training, and synthetic smoke script are not read-only. In particular, `scripts/verify_fusion.py` is a synthetic optimization smoke test that performs gradient updates; it is not a verifier to run under a no-training restriction. `scripts/evaluate_multimodal.py` writes comparison reports.

## Interfaces

- Sensor windows: float tensors shaped `[N, 30, 17]` before the LSTM; extracted sensor embeddings are 64-D.
- Text embeddings: float tensors shaped `[N, 768]`; artifact metadata maps each row to `log_id`, `unit_id`, `cycle`, and `source_split`.
- Fusion adapter: one 64-D sensor vector and one or more 768-D log vectors per canonical window; returns aligned tensors, per-log IDs/entities/severity, split labels, and `has_log`.
- Fusion model: scalar RUL prediction from 64-D and 768-D batches; inclusion of `has_log` is explicit at construction and call time.
- Genuine training loader: requires genuine namespaces, provenance fields, equipment-isolated train/validation assignments, timestamp policy/cutoffs, evidence-referenced approvals, a matching verification report, and a supplied sensor checkpoint. Structural fields do not authenticate the approver or evidence.

## Remaining owners and handoff

Before genuine fusion training, the dataset/privacy owners must provide permissioned paired data and lineage; sensor and technician-system owners must validate equipment mapping, cycle and timestamp semantics; model/evaluation owners must approve target, horizon, split and missing/multiple-log policies; the baseline owner must locate and verify authentic checkpoints; and the project lead plus data/evaluation owners must record explicit training authorization. These responsibilities are roles, not evidence of approval.

IoT integration requires separately approved broker/database configuration, non-production credentials and service tests. No connection details or credentials are present in this repository, and none have been invented.

## Supplemental handoff documents

Some files under `docs/` are pre-existing local handoff materials and remain untracked in this checkout. Two additional proposed documents were added during this review: the fusion evaluation protocol and synthetic-data requirements. Review all documents before integration; older materials may describe earlier branch or environment snapshots. The tracked `context/` files and this README describe the implementation inspected on the current branch.