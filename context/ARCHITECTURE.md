# Implemented Architecture and Readiness

## Scope and status

This repository implements the sensor-only FD001 pipeline, synthetic-capable
technician-log NLP artifact generation, and a metadata-aligned fusion model and
adapter. The fusion components are interface-validated with synthetic data.
Genuine paired-data training and predictive-performance evaluation are blocked
by missing external evidence and authorization. No MQTT, InfluxDB, MongoDB,
Azure, Ollama, spaCy, or service-integration implementation is present in this
checkout.

## Sensor path

```text
data/train_FD001.txt, data/test_FD001.txt, data/RUL_FD001.txt
  -> preprocessing/cmapss_fd001.py or training/dataset.py
  -> engine-isolated train/validation split
  -> StandardScaler fit on training rows only
  -> 30-cycle windows with 17 explicitly selected features
  -> models/lstm_baseline.py
  -> 64-D final LSTM hidden-state embeddings and window metadata
```

The baseline target is uncapped `max_cycle(unit) - cycle`. The test pipeline
creates terminal windows and associates official final RUL labels, but training
and model selection are documented as train/validation only. Do not use the
official test labels for development or modify frozen artifacts. The original
baseline checkpoints are absent from the checked-in model directory; obtain
them from the baseline owner for read-only verification rather than retraining.

Two overlapping preprocessing/data-preparation implementations exist:
`preprocessing/cmapss_fd001.py` writes `artifacts/preprocessing/fd001`, while
`training/dataset.py` prepares the baseline train/validation/test windows under
`artifacts/data/fd001_lstm_baseline`. They share feature constants but produce
different artifact contracts. Existing baseline outputs are frozen; consolidation
requires a separately reviewed migration and must not regenerate them.

## NLP path

```text
technician-log CSV (requires log_id and text)
  -> nlp/preprocessing.py (original and normalized text)
  -> nlp/ner.py (rule-based entities and source offsets)
  -> nlp/severity.py (unvalidated severity heuristic)
  -> nlp/embeddings.py (DistilBERT, 768-D masked mean pooling)
  -> processed logs, entities.jsonl, severity.csv, embeddings.pt, metadata
```

The checked-in five-log data is synthetic. Entity records carry their own
`log_id`; canonical fusion validation rejects missing, unknown, or mismatched
entity ownership. The embedding builder loads model/tokenizer assets with
`from_pretrained` and may download them when invoked if they are not cached.
The fixture tests replace only model inference with deterministic vectors.

## Fusion path

`training/fusion_dataset.py` joins sensor windows and log records by
`(unit_id, cycle)`. It rejects duplicate sensor keys and log IDs, unmatched
records, split inconsistencies, invalid dimensions, non-finite vectors, and
invalid entity ownership. Shapes are `[64]` per sensor record and `[768]` per
technician record. Multiple log embeddings may be mean-aggregated in sorted
`log_id` order, while per-log ownership, entities, severity and source metadata
remain in the result. Missing logs may use a zero vector plus `has_log=False`.
These policy options must be selected explicitly and remain unapproved for
production.

`models/multimodal_fusion.py` concatenates the 64-D and 768-D embeddings and
uses an MLP to output one scalar. `has_log` inclusion is explicit. The output is
an unconstrained RUL prediction. The training harness requires a genuine
approval register and training authorization, but structural validation does
not authenticate approvers or external evidence.

## Artifact and security boundary

The baseline embedding loader hashes consumed sensor embedding bytes and checks
an artifact-local index. The fusion adapter checks supplied artifact hash
syntax and lineage fields, but it does not cryptographically bind a text vector
to approved source bytes or authenticate approval evidence. Training consumes
inline text vectors from JSON and records consumed-file hashes; a trusted,
owner-approved binding from each vector/artifact to those bytes is not yet
defined. The adapter's returned dict/tensors are mutable and have no signed or
content-bound validation receipt. Do not use these mechanisms as a security
boundary for genuine training until a reviewed trust and receipt design exists.

## Infrastructure and deployment

No broker client, time-series database client, document database client,
Azure integration, deployment API, equipment configuration loader, or offline
replay service exists in this checkout. There are no credentials or service
configuration files. These capabilities require separate implementation,
approved interfaces, and isolated service tests; no connectivity is implied by
the model modules.

## Evidence and blockers

- Test suite and artifact verifiers exercise synthetic/interface behavior.
- No approved genuine paired sensor-technician dataset or equipment mapping is
  checked in.
- Genuine cycle relationship, timestamp availability/cutoff, RUL horizon,
  split assignments, missing/multiple-log policy and training authorization
  remain pending.
- Baseline checkpoint files are missing, and dependency pins describe only the
  locally inspected environment.
- No genuine fusion model has been trained or evaluated; no official FD001 test
  evaluation is established by this repository state.
