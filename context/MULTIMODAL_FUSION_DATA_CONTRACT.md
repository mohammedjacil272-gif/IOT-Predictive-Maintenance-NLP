# Multimodal Fusion Data Contract (Provisional)

**Status:** Review draft. This describes the current adapter and proposed
production behavior; it does not approve real sensor/log pairing or training.
See [the approval checklist](MULTIMODAL_APPROVAL_CHECKLIST.md) for decisions
that need team sign-off.

## Fail-closed provenance and policy selection

- Every sensor and text record must state one consistent `namespace`.
  `synthetic_fixture` is isolated from `genuine`; mixed or missing namespaces
  are rejected. Genuine use also requires affirmative approvals for dataset
  permissions, ID mapping, cycle and timestamp semantics, cutoff/horizon,
  uncapped RUL target, split/evaluation protocol, embedding provenance, and
  both log policies.
- Genuine text records require `source_system`, `source_record_id`, and the
  four distinct timestamp fields `event_time`, `record_creation_time`,
  `ingestion_time`, and `availability_time`. Duplicate source IDs and
  ambiguous cross-system collisions are rejected. The approved timestamp
  policy must explicitly select offset-aware timestamp encoding, an IANA
  timezone, cutoff inclusivity, and late-arrival handling. Naive timestamps are
  rejected. Without that approval and cutoff, genuine records fail closed.
  Cutoffs are supplied per prediction window keyed by `(unit_id, cycle)` (or
  its canonical `unit_id:cycle` string); a single shared cutoff is rejected
  for genuine records. Fixture timestamps may only be explicit fixture values.
- **IMPLEMENTED:** genuine sensor and technician records also require source
  dataset/version, source record ID, equipment mapping ID, preprocessing
  version, artifact ID and SHA-256, and embedding model/revision. Genuine
  records missing any required provenance fail closed. Hashes must be supplied
  by actual artifacts or owners; fixtures use clearly labeled test values.
- `build_fusion_samples` requires explicit `missing_log_policy` and
  `multiple_log_policy` arguments. `zero_vector` and `mean_sorted_log_id` are
  available implementation choices, not defaults or production approval.
- `has_log` is **excluded from model inputs unless explicitly selected**.
  Model construction requires `use_has_log=True` or `False`; genuine training
  additionally requires that choice in the approved run manifest. Whether it
  should be a production feature remains **PENDING APPROVAL**.
- Alignment rejects duplicate `(unit_id, cycle)` windows, invalid cycles,
  split conflicts/leakage, malformed or non-finite vectors, duplicate log IDs,
  and logs without a matching sensor window. Sorting/aggregation happens only
  after metadata validation; records are never joined by position.
- The training harness requires an approved run manifest containing schema,
  model/tokenizer names and revisions, preprocessing version, explicit
  operational horizon, policy selections, cutoff and approvals. It also
  requires a passing verification report matching the source commit and
  interpreter, and the owner-supplied sensor checkpoint. Run metadata records
  input hashes, split-manifest identity/hash, model/optimizer/training
  configuration, Python/dependency versions, random seeds, verification
  results and checkpoint lineage. Model/tokenizer checksums are recorded when
  supplied; absent checksums remain null and are never invented.
  A separate evidence-referenced `fusion_training_authorization` is required;
  final held-out evaluation requires its own approval. The harness has not been
  run for genuine training or evaluation.

## Model architecture

The model concatenates each 64-D sensor embedding with its 768-D text
embedding into an 832-D vector. Its configurable MLP regression head is
`832 -> hidden_dim -> ReLU -> Dropout -> head_dim -> ReLU -> 1`, with defaults
`hidden_dim=128`, `head_dim=64`, and `dropout=0.2`. `sensor_dim` and `text_dim`
are explicit model constructor settings; the current dataset contract fixes
them at 64 and 768. The final scalar is an unconstrained RUL prediction.
Dropout is disabled in model evaluation mode; repeated inference in `.eval()`
mode is deterministic for fixed weights and inputs.

## Alignment and labels

- The canonical join key is `(unit_id, cycle)`. Joins must use metadata, never
  array position.
- For sensor records, `cycle` is the source window's `label_cycle`, meaning its
  window-end cycle. The adapter calls this field `cycle`.
- **IMPLEMENTED:** the baseline embedding writer uses non-shuffled inference
  batches and emits sequential `embedding_index` values when concatenating
  arrays. The loader treats this value only as an index into that sensor
  embedding artifact, and validates count, uniqueness, bounds, and complete
  coverage. It is never a sensor/log join key. The currently checked-in sensor
  metadata has this index but lacks required source/artifact lineage fields;
  the genuine loader therefore fails closed on that artifact.
- Shared cycle semantics across sensor and technician systems, and mapping
  real equipment IDs between those systems, remain **PENDING validation on
  genuine paired data**. No mapping may be inferred from the synthetic fixture.
- The fusion adapter consumes the existing sensor record's RUL target unchanged.
  The frozen baseline target is
  `RUL(unit, cycle) = max_cycle(unit) - cycle`, in cycles, without capping.
  Do not alter baseline target generation to support fusion.
- Per-window cutoffs and explicit inclusivity/exclusivity are **IMPLEMENTED**.
  The positive operational horizon must be explicitly configured. Its genuine
  value, cutoff semantics, and timestamp source remain **PENDING APPROVAL**.
  Never infer availability from creation, event, or ingestion time.

## Record fields

### Sensor window record

Required adapter fields: `unit_id` (integer), `cycle` (positive integer,
`label_cycle`/window end), `split` (non-empty string), `sensor_embedding` (one
64-value vector), and `rul` (finite scalar copied from the frozen baseline
label). Duplicate `(unit_id, cycle)` windows and assigning one equipment ID to
multiple splits are rejected. Genuine records additionally require
`source_dataset`, `source_dataset_version`, `source_record_id`,
`equipment_mapping_id`, `preprocessing_version`, `artifact_id`, an actual
artifact SHA-256, and embedding model/revision provenance. Window/source IDs
must come from authoritative metadata.

### Technician log record

Required adapter fields: `log_id` (non-empty, unique string), `unit_id`
(integer), `cycle` (positive integer using the provisional alignment semantics),
`split` (non-empty and equal to the matched sensor record's split), and
`text_embedding` (one 768-value vector). Genuine logs also require source
dataset/version, equipment mapping ID, source system/record ID, artifact and
preprocessing lineage, and the four distinct timestamps. A log without a
matching sensor key, a duplicate log ID, or a split mismatch is rejected.

Optional fields currently accepted by the adapter are `entities`,
`severity_score`, `severity_label`, `matched_terms`, and `explanation`. Supplied
entity provenance is checked against its containing log; output provenance is
retained. Severity is optional and does not replace the RUL target. These
optional fields do not establish source identity or point-in-time availability.

The current NLP artifact builder requires input `log_id` and `text`; it carries
other source columns through preprocessing when supplied. The fusion adapter
also requires equipment, cycle, and split metadata. The real technician-log
pipeline's availability of those fields and their mapping semantics must be
confirmed using permissioned real data.

## Embedding contract

| Modality | Per-record vector | Joined batch | Internal dtype |
| --- | --- | --- | --- |
| Sensor | `[64]` | `[N, 64]` | `torch.float32` |
| Text | `[768]` | `[N, 768]` | `torch.float32` |

The adapter converts inputs to float32, rejects wrong dimensions and non-finite
values, and returns joined tensors. The model accepts matching floating tensor
batches `[batch, 64]` and `[batch, 768]` on the same device and with the same
dtype; it returns one unconstrained prediction per row, shape `[batch]`.
Invalid dimensions must fail validation rather than be padded, truncated, or
silently reshaped.

## Missing and multiple logs

- **PROPOSED / PENDING APPROVAL:** when a sensor window has no matched log,
  retain the window, use a zero 768-D text vector, and set `has_log=False`.
- **PROPOSED / PENDING APPROVAL:** when a window has multiple logs, sort by
  `log_id` deterministically and mean-aggregate their embeddings. Preserve
  per-log IDs, entities, and severity provenance separately.

These are current implementation policies, not approved production choices.

## Split, availability, and leakage requirements

- Train, validation, and test membership is assigned by equipment. An equipment
  ID must not appear across multiple splits. Do not create real pairings or
  splits from the fixture.
- Fusion and sensor-only comparisons must use the same held-out equipment and
  the same frozen target definition. Keep official test labels out of training,
  model selection, and architecture choices.
- The current fusion loader checks split membership using only the
  `source_split` metadata column first. It rejects non-train/validation rows
  before reading RUL targets, and accepts only equipment IDs in the frozen
  train/validation manifest.
  This is a software guard; it does not create or validate genuine pairing.
- A technician log may be an input only if authoritative `availability_time`
  meets that prediction window's cutoff. The adapter enforces this contract;
  genuine timestamp availability and its approved policy remain **PENDING**.
- Future failure information, future-cycle observations, and RUL-derived values
  must never enter model inputs. RUL is the target, not an input feature.
- The synthetic five-record fixture does not establish genuine equipment
  pairing, time availability, split membership, or FD001 test/validation data.
- **IMPLEMENTED:** split manifests are deterministic, include namespace and a
  canonical SHA-256 identity, reject pairwise equipment overlap, and can check
  complete coverage against an expected equipment-ID set. Fixture manifests
  use `synthetic_fixture`; training manifests use `genuine`. A test split
  cannot be declared approved without an evidence-backed approval record.
- The current sensor artifact's `embedding_index` is validated as an
  artifact-local lookup only, never as a cross-modal key. The existing
  artifact lacks an explicit genuine namespace and required source/artifact
  provenance fields and is **BLOCKED** for genuine loading until authoritative
  metadata is supplied.

