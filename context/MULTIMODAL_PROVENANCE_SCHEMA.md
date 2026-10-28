# Multimodal Provenance Schema (Review Draft)

This schema distinguishes fields used by the current adapter from fields that
are optional or not yet available. “Required” describes the current fusion
record contract, not proof that the real source system supplies the value.

The hardened adapter requires one explicit `namespace` on every record and
rejects mixed namespaces. `synthetic_fixture` is only for fixture checks;
genuine records require all approval gates in the decision register. The
canonical candidate key remains `(unit_id, cycle)`, with sensor cycle sourced
from `label_cycle` / window end. No row-order or positional association is
permitted.

| Field | Status | Meaning and current evidence |
| --- | --- | --- |
| `log_id` | REQUIRED | Non-empty unique log-row identifier for fusion input. The NLP builder requires it. It is not necessarily a source-system ID. |
| `unit_id` | REQUIRED | Equipment identifier for fusion alignment. The fixture values are synthetic; real ID mapping is unvalidated. |
| `cycle` | REQUIRED | Integer cycle aligned to sensor `label_cycle`/window end under the provisional contract. Shared real-world semantics remain pending. |
| `window_start_cycle`, `window_end_cycle` | REQUIRED for genuine sensor windows | Must be supplied by sensor metadata; `label_cycle` must match `window_end_cycle`. Bounds are not synthesized. |
| `split` | REQUIRED | Split label carried by the record and matched sensor window. Real membership must follow equipment-level assignment; fixture uses `fixture`. |
| `text` | REQUIRED at NLP input | Raw log text used by the NLP artifact builder. Access, retention, and redaction rules for genuine logs need approval. |
| `text_embedding` | REQUIRED at fusion input | 768-D finite vector; generated artifacts retain embedding row metadata. |
| `source_dataset`, `source_dataset_version` | REQUIRED for genuine | Identify the authoritative source collection and its version. Genuine records without both fail closed. |
| `namespace` | REQUIRED | One consistent namespace per sensor/text dataset; synthetic and genuine namespaces cannot mix. Genuine namespace does not itself prove permission or approval. |
| `source_system` | REQUIRED for genuine | Source namespace for a log identifier; genuine records without it fail closed. |
| `source_record_id` | REQUIRED for genuine | Source-system record identifier for a log or sensor window; missing values fail closed. Duplicate genuine log source IDs and ambiguous collisions are rejected. |
| `equipment_mapping_id` | REQUIRED for genuine | Identifies the owner-supplied equipment mapping used to produce `unit_id`; no mapping is inferred from fixture data. |
| `preprocessing_version` | REQUIRED for genuine | Version of the transformation that produced the record/embedding input. |
| `artifact_id`, `artifact_sha256` | REQUIRED for genuine | Artifact identity and 64-character SHA-256 digest. The run independently hashes files it consumes; an absent digest is not fabricated. |
| `embedding_model`, `embedding_revision` | REQUIRED for genuine | Model and immutable revision used to produce the embedding. Model/tokenizer checksums are recorded when available in the run manifest. |
| `embedding_index` | REQUIRED in the current sensor embedding artifact | Artifact-local lookup only. The baseline writer emits it from deterministic non-shuffled inference output; loader checks uniqueness, row count, bounds, and complete index coverage. It is never a cross-modal key. Existing metadata lacks the source/artifact lineage fields above, so genuine loading fails closed. |
| `source_document_identifier` | OPTIONAL | Preserve when the source provides it. No dedicated field is currently emitted or confirmed. |
| `record_creation_time` | REQUIRED for genuine | Distinct timestamp field. Must be timezone-aware and preserved; semantics/source require approval. |
| `event_time` | REQUIRED for genuine | Distinct timestamp for the represented maintenance event; must not be substituted for availability without approval. |
| `ingestion_time` | REQUIRED for genuine | Distinct source-ingestion timestamp; capture/source semantics require approval. |
| `availability_time` | REQUIRED for genuine; **PENDING evidence** | **Not currently available / pending confirmation.** Required for point-in-time eligibility. Never substitute creation/event/ingestion time without approved equivalence. |
| `timezone_policy` | PENDING | Must explicitly select offset-aware encoding, an IANA timezone, interpretation/normalization and clock assumptions. Naive timestamps are rejected. Runtime validation fails closed without it. |
| `late_arrival_policy` | PENDING | Must explicitly say whether an after-cutoff log is excluded or causes rejection. No default is selected. |
| `checkpoint_lineage` | REQUIRED for a future genuine training run | Owner-supplied source checkpoint ID and SHA-256 when available; fusion training also requires the supplied checkpoint file and independently hashes it. Current frozen baseline checkpoint is missing locally. |
| `operational_horizon` | REQUIRED configuration; **PENDING APPROVAL** | Explicit positive value and unit; the implementation supplies no default or production horizon. |
| `entities` | OPTIONAL | If present on genuine logs, every entity must carry matching `log_id`, `unit_id`, `cycle`, and split provenance; missing or conflicting fields fail closed. Fixture entities remain synthetic. |
| `severity_score`, `severity_label`, `matched_terms`, `explanation` | OPTIONAL | Existing per-log heuristic outputs. They are not ground-truth labels; preserve and validate provenance where supplied. |

The fixture's `generation_rule_id` and `template_id` identify synthetic fixture
construction only. They must not be presented as real source IDs or evidence of
real-world lineage. The fixture does not contain timestamp evidence for log
creation, event occurrence, ingestion, or model availability.

The implementation preserves the four timestamp fields as separate values and
requires an explicit, approved point-in-time policy and a cutoff for every
genuine prediction window, keyed by `(unit_id, cycle)` (or canonical
`unit_id:cycle`). Cutoff inclusivity and late-arrival behavior are explicit
policy values. Presence of synthetic test timestamps is not evidence about
genuine pipeline availability or the production timezone/late-arrival policy.

The current checked-in sensor artifact has an `embedding_index` emitted by the
baseline embedding writer, but its metadata lacks the source dataset/version,
source record ID, mapping ID, preprocessing version, artifact ID and embedding
model revision, and explicit namespace now required by the genuine loader. This artifact is therefore
**BLOCKED for genuine fusion use** until an authoritative artifact/metadata
version is supplied. The existing index is not used to align technician logs.

