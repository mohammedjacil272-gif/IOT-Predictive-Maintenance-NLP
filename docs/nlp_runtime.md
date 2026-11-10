# NLP Runtime and Integration Contract

## Current implementation

The repository provides a local Python pipeline and application-facing
`MaintenanceNLPService` in `nlp/service.py`. This checkout contains no FastAPI
application, route definitions, database models, frontend, or dependency
manifest. The service is the integration boundary available here; a live HTTP
route or frontend workflow has not been implemented or verified.

The service accepts one mapping with a non-empty `log_id` and a non-empty text
string no longer than 20,000 Unicode characters. It returns `log_id`, the
original and normalized text, entity records, severity evidence, an optional
embedding, the embedding model/dimension when available, and an explicit
`embedding_status`. Invalid, empty, and oversized text raises
`InvalidMaintenanceLogError`; callers can map that to HTTP 422 if they add an
API route.

## Preprocessing

Text uses Unicode NFKC normalization, whitespace collapse, trim, and lowercase.
The source text remains unchanged. Character limits apply before transformer
tokenization; DistilBERT truncates token sequences to 128 tokens. Entity offsets
are Python string indexes into the original text, not normalized text.
Industrial terms, codes, units, and measurements are retained by preprocessing.

## Entities

`extract_entities` emits rule-based records with `log_id`, exact `entity_text`,
`label`, inclusive `start`, exclusive `end`, and `rule_source_id`. Supported
labels are `COMPONENT`, `FAULT`, `SYMPTOM`, `ACTION`, `SEVERITY_TERM`, and
`MEASUREMENT`. No confidence value is emitted because these rules have no
calibrated confidence. Equipment IDs, alarm codes, and time references are not
claimed as extracted categories in this version.

## Severity

`score_severity` returns an integer score, label, matched terms with source
offsets, and an explanation. Scores 1 to 4 map to low, medium, high, and critical;
0 means unknown. Low/moderate/severe terms map to their stated levels; common
fault and abnormal-operation indicators map to high; emergency, shutdown, and
safety-hazard terms map to critical. The highest unnegated match wins. A
negation cue within four words in the same clause suppresses that match.
This deterministic heuristic is not an industrially validated risk score or a
probability, and its limited negation scope can misread complex language.

## Transformer embeddings and offline behavior

The embedder uses `distilbert-base-uncased`, matching AutoTokenizer and
AutoModel, a 128-token maximum, attention-mask mean pooling, and 768 output
dimensions. It freezes model parameters, uses evaluation/inference mode,
excludes padding from pooling, returns zero vectors for blank text and an
empty `(0, 768)` tensor for an empty batch, and rejects malformed inputs,
unexpected dimensions, or non-finite output.

Model loading is process-cached. Normal operation sets `local_files_only=True`;
it does not download weights. Set `IOT_NLP_CACHE_DIR` to a Hugging Face cache
root, or a local model directory containing `config.json` and its weights. An operator must explicitly set
`IOT_NLP_ALLOW_DOWNLOAD=1` before process startup to allow model resolution
that may access the network. Model/tokenizer revision hashes are not pinned in
this repository and should be recorded before production use.

If local model weights cannot be loaded, the service still returns normalized
text, rule-based entities, and heuristic severity with
`embedding_status="rule_based_fallback"`, `embedding=null`, and a reason. It
does not fabricate a transformer vector. Successful embedding responses use
`embedding_status="transformer"` and include a 768-value list.

## Fusion compatibility

Sensor windows expose 64-dimensional learned sensor embeddings in their own
artifact interface; technician text embeddings are 768-dimensional. These
vectors are not directly concatenated and this repository does not define a
trained projection or fusion model. `prepare_fusion` aligns logs by
`(unit_id, cycle)` to sensor `(unit_id, label_cycle)`, requires matching
`source_split`, averages multiple log embeddings deterministically, retains
all sensor windows, emits `has_log`, and reports unmatched or unalignable log
IDs. Missing text modality uses a zero 768-vector with `has_log=false`.

No genuine paired sensor/log dataset, authoritative equipment mapping,
point-in-time availability evidence, or fusion training authorization is
established. Fusion training and predictive performance evaluation remain
blocked; the synthetic fixture is for interface checks only.

## Verification status

- **Implemented:** preprocessing, rule-based entities and severity,
  offline-first transformer loader, explicit fallback result, and fusion
  alignment interface.
- **Locally tested:** unit tests and synthetic fixture structure/alignment.
- **Live-service verified:** none; there is no HTTP service in this checkout.
- **Planned:** a FastAPI route and frontend integration when those components
  are present, plus validated genuine paired data before fusion training.

Run `python -m pytest -q` from the repository root for the NLP and fusion unit
tests. Transformer pooling and fallback tests use local test doubles; they do
not establish that external model weights are available. The fixture artifact
builder requires local weights unless model execution is extended explicitly.
