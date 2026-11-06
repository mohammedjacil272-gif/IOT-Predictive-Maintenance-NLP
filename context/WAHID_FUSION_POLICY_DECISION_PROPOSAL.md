# Wahid Fusion Policy Decision Proposal

**Status: PROPOSED — PENDING JESSE APPROVAL**
**Scope:** Deterministic missing-log, multiple-log, and `has_log` decisions for multimodal fusion. This proposal records choices for review; it does not approve or change runtime behavior.
**Reviewed implementation:** current `feature/wahid-multimodal-fusion` at `2a0de7f`.
**NLP reference:** `origin/feature/rafaz-nlp-pipeline` at `f22e8ef` (documentation and validator review; no generated NLP tensor/metadata artifacts present there).

## Decision summary

| Decision area | Recommended option | Status |
|---|---|---|
| Missing logs | Preserve every sensor window; zero 768D text vector; `has_log=False`; preserve missingness and excluded-log provenance separately. | **PENDING JESSE APPROVAL** |
| Multiple logs | Group by `(unit_id, cycle)` after eligibility checks; sort by `log_id`; validate every vector; mean the eligible vectors; retain each log’s provenance and annotations. | **PENDING JESSE APPROVAL** and data-owner confirmation for any derived metadata aggregation |
| `has_log` | Keep as metadata and include as an explicit model feature for an approved model contract. | **PENDING JESSE APPROVAL** |

## Decision 1 — Missing technician logs

### Available options

1. **Zero-vector representation:** preserve each sensor window, emit a 768D zero text vector if no eligible log exists, set `has_log=False`, and retain missingness/provenance explicitly.
2. **Error on missing log:** fail dataset construction if any sensor window has no eligible log.
3. **Drop sensor window:** omit windows without eligible logs. This changes the sensor sample population and can bias it toward better-documented equipment/time periods.

### Current implementation behavior

`build_fusion_samples()` requires an explicit `missing_log_policy` (`zero_vector` or `error`). Under `zero_vector`, each sensor window remains in the result. A window with no eligible text records gets a zero `float32` vector of length 768 and `has_log=False`; a window with at least one eligible record gets `has_log=True`. Missing windows have empty per-window log ID/entity/severity/timestamp/source lists. The current return structure does not provide an enumerated per-window missing-reason field; cutoff-excluded records are preserved separately in top-level `excluded_logs` with their key and reason.

For genuine records, the loader checks approved timestamp policy and per-window cutoff. An after-cutoff log is added to the top-level `excluded_logs` list with its source IDs, timestamps, and exclusion reason, then excluded from that window’s aggregation. If all logs for a sensor window are excluded, the sensor row remains and is represented as missing under `zero_vector`; if some eligible logs remain, those eligible logs alone determine the embedding and `has_log=True`. The exclusion list is separate from eligible per-window log provenance.

### Recommended option

**Option 1: zero vector plus explicit missingness.** Preserve every sensor window; use a 768D zero vector and `has_log=False` when there is no eligible log at the approved prediction cutoff. Retain a distinct missingness reason/state in dataset metadata (`no source log`, `all logs excluded by cutoff`, or other validated ingestion status) and retain every excluded record with its exclusion reason and provenance. Never synthesize a log record to fill a gap. Treat an embedding as available only when at least one eligible, validated log is present.

### Technical rationale

This preserves the frozen sensor-window population and target alignment. The explicit indicator prevents a missing text vector from being interpreted as evidence that an NLP embedding was supplied. Applying cutoff eligibility before aggregation prevents future-available records from leaking into inputs. Distinguishing “no source log” from “logs existed but were unavailable by cutoff” helps audit data coverage without admitting those excluded records into model input.

### Risks and limitations

- A zero vector may also occur as a valid embedding value; the indicator is needed to distinguish that case.
- Missingness can correlate with equipment, maintenance workflow, or reporting quality. A model may learn those patterns, so monitor coverage by equipment and split and review availability bias.
- `has_log=False` alone does not explain why the log is absent; an explicit reason belongs in metadata/provenance, not as an unapproved feature.
- Under `error`, this sample-preserving recommendation does not apply; the policy must be explicit in the approved run manifest.
- A timestamp-excluded log is not a missing source record. It must remain auditable as excluded and must not contribute to the text vector or eligible-log count.

### Approval and input required

- **Jesse approval required:** choose zero-vector versus error behavior and approve explicit missingness/exclusion accounting.
- **Rafaz/data-owner input required:** establish genuine log availability fields and their semantics, authoritative source IDs, per-window cutoff evidence, and a complete record of rejected/unmatched/excluded logs. The remote NLP branch does not supply genuine values or generated artifacts.

## Decision 2 — Multiple technician logs for one sensor window

### Available options

1. **Mean aggregation:** group eligible logs by canonical `(unit_id, cycle)`, sort by stable `log_id`, validate all vectors, and take the arithmetic mean.
2. **Error on multiple logs:** reject any window with more than one eligible log.
3. **Alternative learned/weighted aggregation:** requires a separately specified and approved method and evidence; none is proposed as the current choice.

### Current implementation behavior

The adapter requires explicit `multiple_log_policy` (`mean_sorted_log_id` or `error`). It validates unique non-empty `log_id`, matching sensor key and split, 768D shape, and finite values; then sorts eligible records by `log_id`. The mean is computed over the sorted vectors. It preserves per-window `log_ids_by_window`, per-log entities, per-log severity records, timestamps, source IDs, and provenance. Entities are flattened in sorted-log order, but remain records carrying their originating `log_id`. `window_severity` is currently a derived maximum of available numeric severity scores; individual scores remain in `severity_by_log_by_window`.

### Recommended option

**Option 1: deterministic arithmetic mean of eligible log embeddings.** Group only after key, split, duplicate-ID, provenance, timestamp, dimension, and finite-value checks. Sort by stable `log_id`; average all eligible 768D embeddings using the existing `mean_sorted_log_id` implementation. Preserve the complete ordered per-log records and provenance beside the aggregate. Keep `error` as an explicit alternative; never make aggregation an implicit default.

For entities and severity, do not collapse the source records into a replacement “aggregate log.” Preserve all entities and per-log severity in stable log order with each `log_id`. A derived `window_severity=max(scores)` may be retained as a clearly labeled summary only if owners approve its semantics; it is lossy, and heuristic NLP severity must not be treated as ground truth. No entity deduplication, severity weighting, or cross-log entity merge is recommended without an explicit rule and approval.

### Technical rationale

Stable grouping makes results independent of input row order and preserves one training row per sensor window. Mean pooling keeps the text input at 768 dimensions regardless of log count. Per-log provenance maintains traceability and allows later review of which eligible logs contributed. Individual annotation retention avoids silently erasing disagreement or repeated evidence.

### Risks and limitations

- The mean can dilute a short, high-information log among many routine logs and can conceal contradictory notes.
- `log_id` ordering makes execution and provenance deterministic; it does not assign clinical/maintenance importance.
- Duplicate records with different IDs are not necessarily detectable without authoritative source-record identity and deduplication rules.
- The current `window_severity` maximum is a lossy summary and can amplify heuristic false positives. It is not the embedding aggregation rule and is not proposed as a model input.
- Mean aggregation applies only to eligible logs after cutoff. Excluded logs remain in audit metadata and do not affect the mean.

### Approval and input required

- **Jesse approval required:** approve mean versus error and whether multiple-log policy is fixed or remains a manifest-selectable approved option.
- **Rafaz/data-owner input required:** confirm stable genuine `log_id`, source-level duplicate semantics, and whether source annotations are retained per log. Any requested entity/severity aggregation requires a separate specified rule. The current NLP documentation proposes mean embeddings but explicitly leaves entity/severity aggregation for confirmation.

## Decision 3 — `has_log` model-input policy

### Available options

1. **Metadata-only:** preserve the boolean for audit, stratification, and coverage reporting, but do not concatenate it into model input.
2. **Model-input feature:** append the boolean as one numeric feature in addition to the metadata copy.

### Current implementation behavior

`MultimodalFusionModel` requires an explicit `use_has_log=True` or `False`; omitting it raises an error. With 64D sensor and 768D text embeddings, excluding the flag yields an 832-feature concatenated input to the first linear layer. Including it yields 833 features. The forward method requires a batch-shaped `has_log` tensor when enabled. `training/train_multimodal.py` requires an explicit CLI selection and requires it to match the approved run manifest. It does not silently select a production policy.

### Recommended option

**Option 2: retain as metadata and include as an explicit model feature for a future approved model contract.** Keep the include/exclude switch and record the approved choice in the run manifest; do not select it through code defaults. If Jesse chooses metadata-only, continue retaining the flag in joined dataset outputs for auditing.

### Technical rationale

When missing text is encoded as a zero vector, the indicator lets the model distinguish missing text from an available embedding whose values happen to be zero. It also lets the model learn whether text availability changes the usefulness of other inputs. Keeping it as metadata in either option supports coverage and missingness analysis.

### Risks and limitations

- Including it changes the first layer input width from 832 to 833 and changes the model parameterization/checkpoint contract. Existing trained weights cannot be assumed compatible.
- The indicator may encode documentation or workflow patterns and can become a shortcut correlated with target or split. Availability bias requires review.
- A feature decision based on synthetic fixtures cannot establish genuine predictive utility. No predictive-performance evaluation is authorized by this proposal.
- If cutoff-excluded logs and never-recorded logs are both represented by `has_log=False`, the model does not see the distinction unless an additional feature is separately proposed and approved; keep the reason in metadata.

### Approval and input required

- **Jesse approval required:** choose metadata-only or model-input; current recommendation is model-input, pending approval.
- **Model/evaluation owner input required:** confirm the architecture/checkpoint versioning implications and review missingness leakage risk before any approved training.
- **Rafaz/data-owner input required:** provide genuine availability semantics and coverage evidence; no such evidence or embeddings are present on the reviewed remote branch.

## Rafaz NLP-to-fusion field mapping

The reviewed remote branch is `origin/feature/rafaz-nlp-pipeline` at `f22e8ef`. It contains docs/code/tests and the synthetic CSV input fixture, but no generated NLP embedding tensor or embedding metadata. These are proposed adapter mappings only; they do not fabricate or assert genuine values.

| Fusion record field | NLP source field or mapping | Status / action |
|---|---|---|
| `log_id` | `embedding_metadata.csv.log_id` / processed-log `log_id` | Stable fixture ID; unique. Must map to authentic source records in genuine data. |
| `text_embedding` | `embeddings.pt[row_index]` where `row_index` is metadata’s artifact-local index | Row map is checked in branch verifier for synthetic outputs. The actual generated artifacts are absent from the branch. |
| `unit_id` | Metadata/processed `unit_id` | Optional in Rafaz builder. Genuine ID/crosswalk and fusion `equipment_mapping_id` are not supplied; BLOCKED. |
| `cycle` | Metadata/processed `cycle` | Optional and semantics unverified. Proposed sensor correspondence is `label_cycle`; mapping requires owner evidence. |
| `split` | NLP `source_split`, after explicit validation/conversion | Fusion requires `split` and exact match to sensor split. Fixture value `fixture` is not a genuine train/validation assignment. |
| `namespace` | Per-record value derived only from verified artifact classification (`synthetic_fixture` for fixture) | NLP report has dataset-level `dataset_type`; fusion requires record namespace. Genuine namespace cannot be assigned without genuine source evidence. |
| `source_dataset`, `source_dataset_version` | No established source fields | Required by genuine fusion; obtain from data owner, do not infer. |
| `source_system`, `source_record_id` | Documented unavailable/unconfirmed | Required by genuine fusion; obtain from source system/owner. `log_id` is not a substitute unless owner establishes that identity relation. |
| `equipment_mapping_id` | No supplied mapping artifact/field | Required for genuine fusion. Obtain authoritative versioned crosswalk and evidence. |
| `preprocessing_version` | No version recorded by current NLP artifacts | Required for genuine fusion; assign only through approved release metadata. |
| `artifact_id`, `artifact_sha256` | No generated NLP artifact payload/manifest hash in branch | Required by genuine fusion; calculate from delivered artifact and record actual identity. |
| `embedding_model`, `embedding_revision` | `model_name` (`distilbert-base-uncased`) maps only to model name | 768D dimension is compatible. Exact immutable model revision/checksum is absent and must be supplied, not guessed. |
| Tokenizer name/revision | Not fully recorded in embedding metadata | Run-level reproducibility contract requires both. Obtain exact tokenizer revision/checksum. |
| `event_time` | Candidate source: `event_timestamp` | Meaning, timezone, offset-aware values and genuine source are unverified; requires owner-approved mapping. |
| `record_creation_time` | Candidate source: `created_at` | Same evidence/semantics gap. |
| `ingestion_time` | Candidate source: `ingested_at` | Same evidence/semantics gap. |
| `availability_time` | Candidate source: `available_at` | Candidate semantic mapping only. Fusion uses it for cutoff eligibility; trustworthy source, timezone, late-arrival rule and per-window cutoff remain unapproved/unavailable. |
| `entities` | `entities.jsonl`, grouped by `log_id` | Fixture entity output exists by manifest but payload absent from remote tree. Preserve log provenance; no cross-log deduplication rule approved. |
| `severity_score`, `severity_label`, `matched_terms`, `explanation` | `severity.csv` fields | Name mapping required (`severity_matched_terms`/`severity_explanation` variants occur in processed output). Scores are heuristic; do not treat as ground truth. Payload is absent from remote tree. |

The adapter must validate actual metadata row counts, unique and in-range row indexes, unique log IDs, finite 768D vectors, field consistency, and source provenance before constructing fusion records. All genuine source, mapping, timestamp, model-revision, and hash fields remain **BLOCKED** until authoritative input is supplied.

## Approval record

No approval is granted by this document. Record Jesse’s decision and evidence in the approved versioned fusion policy/run manifest before any genuine-data use. Any change to model input shape also requires explicit model-contract versioning. Rafaz/data-owner confirmations must be attached for source identity, mapping, cycles, timestamps, and artifact lineage.
