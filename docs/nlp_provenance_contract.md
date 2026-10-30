# NLP Provenance and Data Alignment Contract

**Status:** Version 1.0 — implementation-aligned draft
**Owner:** Rafaz
**Scope:** Technician-log processing and future sensor/NLP integration

## 1. Purpose

Define how technician logs are identified, processed, traced and prepared for possible integration with sensor windows.

This document distinguishes currently implemented behaviour from future requirements that need authoritative data and project approval.

## 2. Current provenance

The current pipeline:

1. Reads technician logs from CSV.
2. Requires non-empty, unique log_id values.
3. Preserves original input fields.
4. Preserves original_text.
5. Generates normalized_text for embedding and severity processing.
6. Extracts entities from original text.
7. Generates heuristic severity metadata.
8. Generates 768-dimensional text embeddings.
9. Writes embedding metadata with row_index and log_id.
10. Writes a verification report identifying the dataset as synthetic_fixture.

Current fixture:
`tests/fixtures/technician_logs.csv`

Current output locations:

- `artifacts/data/fd001_nlp_fixture/`
- `artifacts/embeddings/fd001_nlp_fixture/`

These are fixture-specific locations.

## 3. Stable identity requirements

Every genuine source log must have:

- A stable, unique log_id.
- Its original source record identifier where available.
- Its source system.
- Its original asset identifier.
- An authoritative mapping to canonical unit_id before sensor alignment.
- A documented cycle or timestamp association.

Missing log_id values must be rejected and reported.

Duplicate log_id values must be detected and reported. They must not be silently overwritten.

Missing unit_id or cycle information must not be fabricated. Such records may remain available for text-only processing but must be marked unalignable for sensor fusion.

## 4. Embedding-to-log mapping

Every embedding row must map deterministically to a log_id.

Current metadata includes:

- row_index
- log_id
- unit_id
- cycle
- source_split
- model_name
- max_length
- pooling
- embedding_dim

Future genuine-data validation must check:

- Tensor row count equals metadata row count.
- Each row_index is unique and valid.
- Every embedding row has a valid log_id.
- Every log_id maps to the intended source record.
- Duplicate or missing mappings are reported.
- Embeddings are finite and have the expected dimension.
- Model and tokenizer revisions are recorded.

Never use CSV row order as a substitute for equipment/cycle identity.

## 5. Sensor alignment contract

The provisional integration contract uses:

`(unit_id, cycle)`

The NLP cycle is mapped to sensor metadata `label_cycle`.

Requirements:

- Use verified canonical equipment identifiers.
- Use an authoritative cycle definition.
- Require matching source_split values.
- Never join by CSV row position.
- If timestamps replace cycles, obtain an approved timestamp-to-window mapping first.
- Preserve unmatched records in data-quality reports.

This is a provisional interface contract, not proof that genuine paired data currently exists.

## 6. Multiple logs per sensor window

The provisional implementation contract proposes:

1. Group logs by `(unit_id, cycle)`.
2. Sort deterministically by log_id.
3. Average the 768-dimensional embeddings for the window.

Aggregation of entity and severity metadata still requires confirmation.

The proposed embedding aggregation must not be treated as final production approval until the project/data owners approve it.

## 7. Missing log policy

The provisional integration contract proposes:

- Preserve every sensor window.
- For a sensor window without a matching log, set `has_log=False`.
- Use a zero vector of length 768 for the missing text embedding.
- Report NLP records that do not match sensor windows.
- Do not silently discard malformed, duplicate or unalignable records.

Whether `has_log` is metadata-only or a model input requires an explicit decision.

## 8. Point-in-time availability and leakage

Genuine records must have trustworthy availability information before they can be considered for prediction-time fusion.

Required source fields to establish:

- event_timestamp
- created_at
- ingested_at
- available_at
- timezone
- cycle/window association

The data owner must document what each timestamp means and provide evidence for its source.

A record must not be used for a prediction if it became available after the approved prediction cutoff.

Do not infer availability from event time, file modification time or row order.

Do not use future maintenance outcomes, RUL labels, terminal cycles or official test labels to construct or select NLP inputs.

## 9. Synthetic and genuine data separation

The current fixture is synthetic.

Synthetic records may be used for:

- Interface tests.
- Schema tests.
- Deterministic join tests.
- Missing-data tests.
- Embedding shape and integrity tests.

Synthetic records must not be:

- Relabelled as genuine.
- Used to establish real equipment mappings.
- Used to claim real-world severity or NER accuracy.
- Used to claim predictive improvement.
- Mixed silently with genuine training data.

Every dataset and artifact must have an explicit origin classification.

## 10. Provenance fields required for genuine data

The following fields are not currently established and require source-owner confirmation:

- Source system and source version.
- Stable source record identifier.
- Original asset identifier.
- Canonical equipment crosswalk and its version.
- Cycle definition and its evidence.
- Event, creation, ingestion and availability timestamps.
- Timezone.
- Permission and approved-use evidence.
- Source dataset hash.
- Model and tokenizer revisions.
- Preprocessing version.
- Artifact lineage.

## 11. Validation report requirements

For each supplied genuine dataset, report:

- Total source records.
- Accepted records.
- Rejected records.
- Missing log IDs.
- Duplicate log IDs.
- Missing equipment mappings.
- Unalignable records.
- Missing cycle information.
- Missing availability timestamps.
- Invalid timestamps or timezone values.
- Embedding rows and metadata rows.
- Missing or duplicate embedding mappings.
- Non-finite embeddings.
- Synthetic/genuine classification.

Do not claim validation PASS when required source evidence is unavailable.

## 12. Approval gates

Before genuine fusion training or predictive evaluation, obtain:

1. Permissioned genuine paired sensor-technician data.
2. Authoritative eligible-equipment manifest.
3. Verified equipment mapping.
4. Approved cycle or timestamp semantics.
5. Trustworthy available_at timestamps.
6. Approved target, prediction cutoff and horizon.
7. Approved equipment split and evaluation protocol.
8. Decisions on missing logs, multiple logs and has_log.
9. Required baseline artifacts and their original provenance.
10. Explicit training and evaluation authorization.

Structural validation of an approval record does not independently authenticate its evidence.

## 13. Baseline protection

Do not modify or retrain the frozen sensor-only FD001 baseline.

Do not recreate missing checkpoints or use official FD001 test labels for model selection or evaluation.

If original baseline checkpoints are unavailable, report baseline verification as BLOCKED.

## 14. Status vocabulary

- PASS: Evidence and required validation are available.
- BLOCKED: A required dependency or evidence is missing.
- PENDING APPROVAL: A decision or authorization is outstanding.
- NOT RUN: The activity has not been performed.
- SYNTHETIC-ONLY: Validation used synthetic fixtures only.

## 15. Current status

The current NLP implementation has been exercised with a five-record synthetic fixture.

Genuine paired data, authoritative equipment mapping, cycle semantics and point-in-time availability are not established by the current fixture.

Fusion training and predictive-performance evaluation remain unauthorized until all required gates are satisfied.
