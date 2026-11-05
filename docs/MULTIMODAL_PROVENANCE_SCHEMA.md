# Multimodal Provenance Schema

**Status:** Proposed production schema. Audit SHA: `1ffc5be56b697a6fe3c97df0c32c1b5ed0f5bf3f`. The current synthetic fixture does not populate all fields. Do not fabricate fields or alter shared interfaces before coordination with Wahid and the data owner.

| Field | Current status | Proposed meaning and checks |
|---|---|---|
| `log_id` | Required fusion input | Unique processed log-row ID; not necessarily the source-system primary key. Preserve a mapping to source IDs. |
| `source_system` / namespace | Not required; pending | Identify authoritative system/tenant/namespace. |
| `source_record_id` | Not required; pending | Stable ID of the raw source record/work order. Never invent one and present it as source provenance. |
| `source_document_id` | Not required; pending | Stable ID of note/document, if distinct. Confirm with owner. |
| `dataset_id`, `dataset_version` | Not required; pending | Approved source release/snapshot and permission scope. |
| `unit_id` | Required integer alignment key; real mapping pending | Canonical project equipment ID after documented mapping. Preserve source equipment ID and mapping version separately. |
| `source_unit_id` | Not required; pending | Original identifier from source, when different. |
| `equipment_mapping_version` | Not required; pending | Version/hash/reference to approved crosswalk. |
| `cycle` | Required positive integer; source semantics pending | Cycle aligned to sensor `label_cycle`/window end under the provisional contract. Record conversion rule. |
| `window_start_cycle`, `window_end_cycle`, `label_cycle` | Existing sensor metadata | Preserve sensor window bounds and label cycle; define how log records relate to the window. |
| `split` / `source_split` | Basic field required | Approved equipment-level assignment. Keep fixture isolated as `fixture`. |
| `event_timestamp` | Not required; pending source semantics | Time the event occurred, if the source defines it. |
| `created_at` | Not required; pending source semantics | Source record creation time; may not prove prediction-system availability. |
| `ingested_at` | Not required; pending | Ingestion time; not automatically equivalent to availability. |
| `available_at` | **BLOCKER** | Authoritative earliest time the record was available to the prediction system. Must be supported by source evidence. Reject if later than prediction cutoff. Never substitute another timestamp without approval. |
| `prediction_cutoff` / `window_cutoff_at` | Not in current contract; pending | Prediction timestamp and timezone/inclusive-boundary rule. May be sample metadata if recorded unambiguously. |
| `text` | Required at NLP builder input | Raw technician text. Approve access, retention, minimization, redaction and audit policy. |
| `text_embedding` | Required fusion input | Finite 768-D vector; trace to source text row and exact embedding-generation configuration. |
| `embedding_model_name` | Current loader says `distilbert-base-uncased` | Name alone is not enough for reproducibility. |
| `embedding_model_revision` | Missing from current loader/report evidence | Immutable model and tokenizer revision selected by NLP owner. |
| `embedding_artifact_sha256` / model asset hashes | Missing | SHA-256 manifest for model/tokenizer files, generated embeddings, and relevant config. |
| `preprocessing_version` | Not part of required fusion input | NLP normalization/truncation/pooling code version. |
| `source_split` provenance | Basic split is implemented; lineage pending | Record how equipment-level assignment was made and version the split manifest. |
| `data_origin` | Not uniformly required | Explicit `synthetic_fixture` vs. `genuine_approved` propagated through all artifacts. Never relabel fixture rows. |
| `permission_reference` / licence | Not in interface; pending | Approved data-use authority/reference, without storing secrets. |
| `entities` | Optional | Per-log entities; supplied parent IDs/metadata checked. Fixture entities are synthetic. |
| `severity_score`, `severity_label`, `matched_terms`, `explanation` | Optional | Heuristic outputs, not ground truth; preserve version and provenance. |
| `ingestion_run_id`, `artifact_build_id` | Not required; pending | Trace extraction/embedding run to outputs and validation results. Processing provenance is not source identity. |

## Proposed minimum production provenance (PENDING APPROVAL)

1. Source identity: dataset/version, source system, source record/document ID, permission reference.
2. Equipment mapping: source ID, canonical `unit_id`, mapping version, cycle semantics and sensor window boundaries.
3. Time eligibility: timestamp names/semantics/timezone; authoritative `available_at`; prediction cutoff and a tested inclusion rule.
4. Split and target: source split, versioned engine split manifest; RUL definition/version in labels only; do not leak it into inference features.
5. Embedding lineage: input log row ID, text preprocessing version, immutable model/tokenizer revision, max length, pooling method, dependencies, asset/file hashes and artifact build ID.
6. Origin/quality: synthetic-vs-genuine tag, missing-text status, validation findings, and source-to-fused-sample lineage.

## Digest limitations

SHA-256 detects byte differences against a known digest; it does not authenticate the origin, prove labels/timestamps true, or show that a technician note existed by the prediction cutoff.

## No-fabrication rule

If the genuine source cannot provide a source ID or reliable availability time, mark it unavailable. Keep performance evaluation blocked or exclude records under a pre-approved rule. Never create substitute timestamps or fabricate a mapping between FD001 and OIL-039.
