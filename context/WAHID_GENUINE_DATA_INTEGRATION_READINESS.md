# Genuine Sensor + Technician Log Integration Readiness

**Assessment:** genuine-data integration is **BLOCKED**. The repository implements useful validation and fusion controls, but the NLP package path is synthetic-only, genuine source mappings and owner approvals are outstanding, and the original FD001 checkpoints are absent. This report documents code and artifacts present in this worktree; a passing fixture test is not evidence of real-world pairing, authorization, or readiness.

## Implemented behavior

### Join and embedding contract

- The canonical cross-modal join key is **`(unit_id, cycle)`**. The sensor cycle is its `label_cycle` / window-end cycle. Fusion does not join technician logs to sensor vectors by row order.
- **`embedding_index` is only a lookup into the sensor embedding artifact.** `training/train_multimodal.py` validates its count, uniqueness, bounds, and complete coverage before indexing the sensor array. It is not an equipment/log key.
- `training/fusion_dataset.py` checks finite 64D sensor and 768D text vectors; positive integer IDs/cycles; unique sensor keys and log IDs; split agreement and unit-level split consistency; duplicate source identity for genuine records; entity parent ownership; and unmatched text records. Logs are sorted by `log_id` before the explicitly chosen deterministic aggregation. Per-log IDs, entities, severity, timestamps, and provenance are retained.
- Missing-log choices (`zero_vector` or `error`) and multiple-log choices (`mean_sorted_log_id` or `error`) are caller-supplied, with no implicit default in the builder. These are implemented choices, **not Jesse-approved production policies**. Model construction likewise requires an explicit `use_has_log` decision.

### NLP package validation and orchestration

- `training/nlp_artifact_validation.py` checks required package files, schema and metadata values, `N × 768` finite floating embeddings, row-index coverage, metadata/processed-log ID and unit/cycle/split consistency, related entity/severity IDs, duplicate records, resource bounds, paths, upstream report consistency, and SHA-256 manifest values when supplied. Unsigned hashes/reports do not prove authenticity.
- `training/nlp_fusion_adapter.py` accepts synthetic fixture inputs only. It requires `source_split="fixture"`, a synthetic fixture report, sequential in-range row indices, exact row-to-log identity, and consistent embedding configuration. It fixes output namespace to `synthetic_fixture`; it does not infer genuine provenance, mappings, timestamps, or approvals.
- `training/nlp_fusion_pipeline.py` validates and loads the complete package before calling the adapter, then passes records to `build_fusion_samples`. It forwards the explicit missing/multiple-log choices and the approval/time-policy arguments. This is a synthetic package integration route, not a supported genuine ingestion route.
- `training/fusion_provenance.py` implements structural approval-register checks, timestamp parsing requiring timezone offsets, and point-in-time cutoff evaluation. `training/fusion_splits.py` builds deterministic equipment-level manifests, rejects overlap/duplicates and can enforce coverage; test assignments require an evidenced approval.

## Readiness by area

| Area | Status | Evidence / gap |
|---|---|---|
| Synthetic NLP artifact validation and adapter | **Implemented; synthetic only** | Validator, adapter, and pipeline tests exercise fixture packages. This does not validate genuine source identity or data pairing. |
| Canonical join and dimensions | **Implemented** | `(unit_id, cycle)` join; 64D sensor and 768D text checks. No array-position join. |
| Entity ownership | **Implemented** | Every persisted entity must carry its own nonempty `log_id`, match its parent log, and not conflict with supplied unit/cycle/split metadata. |
| Multiple/missing-log mechanics | **Implemented; policy approval pending** | Explicit choices and deterministic aggregation exist. Jesse must select the production behavior. |
| Provenance and point-in-time mechanics | **Partially implemented** | Fusion checks required field shapes, timezone-aware timestamps, policy structure, and per-window cutoffs. Sources, timestamp meanings, and approvals are not verified against real systems. |
| Equipment crosswalk and cycle mapping | **Unverified / blocked** | No authoritative source-asset-to-`unit_id` crosswalk or approved technician-cycle-to-sensor-`label_cycle` mapping is present. |
| Genuine NLP intake | **Blocked** | Adapter rejects non-fixture artifacts; no trusted genuine artifact classification, signed/otherwise governed lineage, or genuine payload handoff is established. |
| Genuine sensor lineage | **Blocked** | Current sensor metadata has an artifact-local index but lacks the genuine namespace and required source/artifact provenance used by the genuine loader. |
| Genuine paired records and split assignment | **Blocked** | No permissioned paired sensor/log dataset with validated equipment-level assignments and source evidence is documented here. |
| Training readiness | **Blocked** | Missing original baseline checkpoints and approvals; no training was performed for this report. |

## Genuine-data handoff requirements

### Jesse / authorized project approvers

Provide a versioned approval register with approver identity and evidence references for:

- Dataset permission, use, retention, and privacy constraints.
- Equipment-ID mapping and collision policy.
- Cycle semantics and pairing rule.
- RUL target definition and operational horizon (value and unit).
- Timestamp semantics, timezone/clock assumptions, prediction cutoff and inclusivity, and late-arrival action (`exclude` or `reject`).
- Equipment-level split and evaluation protocol, including a protected held-out set.
- Missing-log and multiple-log policies.
- Whether `has_log` is model input or audit metadata only.
- Embedding provenance.
- A separate `fusion_training_authorization`; final held-out evaluation requires a separate authorization.

The required genuine approval keys are defined in `training/fusion_provenance.py` (`GENUINE_APPROVALS`). Structural validation of an approval file cannot authenticate the approver or prove its evidence is truthful.

### Jesse's data handoff acceptance checklist

- [ ] Identify the authoritative, permissioned sensor and technician-log datasets, versions, owners, access conditions, and retention/redaction limits.
- [ ] Deliver actual paired records and document stable `log_id`, source-system ID, and source-record ID semantics. Do not derive missing IDs from row order or filenames.
- [ ] Supply the versioned source asset → canonical `unit_id` crosswalk, mapping ID, owner, and collision/coverage report.
- [ ] Define technician cycle meaning and provide evidence-backed mapping to sensor `label_cycle` / window end, including boundary examples and unmatched-record handling.
- [ ] Supply genuine split assignments by equipment and verify no equipment overlap. Do not infer splits from fixture labels.
- [ ] Provide the four distinct source timestamps: `event_time`, `record_creation_time`, `ingestion_time`, `availability_time`; define each source field and its clock/timezone basis. Do not substitute one timestamp for another.
- [ ] Approve per-window prediction cutoffs keyed by `(unit_id, cycle)`, cutoff inclusivity, and late-arrival behavior.
- [ ] Decide missing-log, multiple-log, and `has_log` behavior, and record the approved choices.
- [ ] Approve RUL target/horizon and the evaluation/split protocol.
- [ ] Deliver a complete genuine artifact package: `embeddings.pt`, `embedding_metadata.csv`, `processed_logs.csv`, `entities.jsonl`, `severity.csv`, `verification_report.json`, and artifact manifest. Include stable row-to-record mappings, 768D finite embeddings, model/tokenizer immutable revisions, preprocessing version, artifact IDs, and matching hashes.
- [ ] Identify the authentic baseline checkpoint location and provide owner-confirmed hashes and lineage; do not recreate missing checkpoints.
- [ ] Record Jesse/authorized approvals with evidence references, including training authorization only when a later training request is reviewed.

### Rafaz / NLP and source-system owners

- Deliver the actual intended NLP artifacts and an explicit, versioned genuine schema; current checked-in NLP payloads under `artifacts/.../fd001_nlp_fixture` are fixtures.
- Confirm `log_id` semantics and exact mapping of tensor `row_index` to metadata and processed source record.
- Provide source system, source record ID, source dataset/version, original asset identifier, genuine split, model/tokenizer names and immutable revisions, preprocessing version, pooling/max-length configuration, artifact identity and hashes.
- Document authoritative timestamp column mappings and source semantics, especially trustworthy `availability_time`.
- Supply entities with each entity’s own matching `log_id`; when genuine fusion metadata is provided, entity `unit_id`, `cycle`, and split must also agree with its parent log.

## Frozen FD001 baseline

The checked-in baseline metrics are from `artifacts/metrics/fd001_lstm_baseline/validation_metrics.json`:

| Metric | Recorded value | Scope |
|---|---:|---|
| MAE | 20.884394122374314 | Validation |
| RMSE | 31.364907230543583 | Validation; best epoch 3 |
| MSE | 983.7574055806052 | Validation |
| R² | 0.733277969832774 | Validation |
| Official FD001 test evaluation | Not performed | Explicitly recorded as not performed |

Baseline data, embeddings, metrics, and model configuration are under:

- `artifacts/data/fd001_lstm_baseline/`
- `artifacts/embeddings/fd001_lstm_baseline/`
- `artifacts/metrics/fd001_lstm_baseline/`
- `artifacts/models/fd001_lstm_baseline/`

The original `artifacts/models/fd001_lstm_baseline/best_model.pt` and `last_model.pt` checkpoints are **missing**. The configuration files in that directory are not checkpoints. The baseline owner must recover or identify the approved storage location and supply authentic files with lineage and hashes. **Do not retrain, recreate, or modify the frozen baseline** to satisfy this dependency.

## Tests required before any genuine-data training request

First, add or adapt tests only after owners supply the approved schema and mappings. The required pre-training suite should verify:

1. Genuine artifact schema, classification/approval gate, actual hash comparison, safe tensor limits, and rejection of missing or conflicting provenance.
2. Row-index-to-log identity and entity-parent ownership; duplicate IDs, duplicate source IDs, and unit/source-ID collision handling.
3. Asset crosswalk coverage/collisions and cycle-to-`label_cycle` boundaries using owner-supplied examples.
4. Timestamp source mapping, timezone/offset validation, per-window cutoff inclusivity, late-arrival reject/exclude behavior, and audit retention of excluded logs.
5. Canonical `(unit_id, cycle)` alignment, split equality/leakage, unmatched rows, dimensions, finite values, deterministic multi-log aggregation, and explicit missing-log behavior.
6. Genuine package → validator → adapter → fusion integration tests proving validation precedes ingestion and no genuine metadata is inferred.
7. Model input and `has_log` contract checks plus run-manifest authorization, exact split manifest, baseline checkpoint lineage, and verification-report gates.
8. Existing sensor-only baseline protection tests and full project tests on a pinned environment before requesting training authorization.

Tests must not treat synthetic fixtures as genuine evidence. Use synthetic tests for mechanics; verify genuine acceptance only with owner-approved, permissioned data or controlled owner-supplied schema evidence.

## Risks, assumptions, and unresolved decisions

- The NLP validator and adapter currently accept synthetic fixture classification only. A separate, reviewed genuine adapter/contract is required; changing the dataset label alone cannot safely enable genuine ingestion.
- Existing package validation checks consistency and hashes when available. An unsigned manifest or report is not proof of authenticity; the genuine handoff needs a trusted owner-controlled provenance/signing or equivalent verification process.
- NLP timestamp names documented in the artifact schema (`event_timestamp`, `created_at`, `ingested_at`, `available_at`) do not have an approved mapping to fusion's four semantic timestamp names. No mapping is assumed here.
- Matching unit/cycle values and a valid-looking ID do not prove that a technician log belongs to a sensor window. Owners must establish that relationship.
- `zero_vector`, sorted mean aggregation, and either `has_log` choice are implementation choices, not Jesse's approvals.
- Metric values above are recorded validation results. They do not establish performance on an official test set or indicate multimodal benefit.
- This report is a code/artifact review, not an authenticity audit of external data or an approval to train.

## Inspected files

- `training/fusion_dataset.py`
- `training/nlp_artifact_validation.py`
- `training/nlp_fusion_adapter.py`
- `training/nlp_fusion_pipeline.py`
- `training/fusion_provenance.py`
- `training/fusion_splits.py`
- `training/train_multimodal.py`
- `tests/test_fusion_alignment.py`
- `tests/test_fusion_provenance.py`
- `tests/test_nlp_artifact_validation.py`
- `tests/test_nlp_fusion_adapter.py`
- `tests/test_nlp_fusion_pipeline.py`
- `tests/test_nlp_fusion_end_to_end.py`
- `tests/test_nlp_fusion_integration.py`
- `tests/test_training_contract.py`
- `artifacts/metrics/fd001_lstm_baseline/validation_metrics.json`
- `artifacts/metrics/fd001_lstm_baseline/training_history.csv`
- `artifacts/models/fd001_lstm_baseline/model_config.json`
- `artifacts/models/fd001_lstm_baseline/training_config.json`
- `context/MULTIMODAL_FUSION_DATA_CONTRACT.md`
- `context/MULTIMODAL_APPROVAL_CHECKLIST.md`
- `context/MULTIMODAL_PROVENANCE_SCHEMA.md`
- `context/WAHID_MULTIMODAL_INTEGRATION_READINESS_AUDIT.md`
- `context/WAHID_INTEGRATION_EXECUTION_REPORT.md`
