# NLP implementation report

The rule-based technician-log path is implemented and tested: normalization,
negation preservation, entity extraction, severity scoring, missing-log policy,
provenance, and explicit synthetic/genuine classification. The reusable
artifact validator checks dimensions, finite values, row alignment, timestamps,
manifest hashes, resource limits, and path safety.

JSON and JSONL object parsers now reject duplicate keys instead of allowing the
standard library to silently keep the last value. Manifest coverage is required
for every supplied package file that participates in validation.

Transformer embeddings are **unavailable** in this environment: the compatible
package/model revision and genuine approved log corpus were not available, and
the pipeline does not download weights at startup. No transformer metrics or
embeddings are claimed.

Relevant checks: `scripts/verify_nlp.py`, `tests/test_nlp_artifact_validation.py`,
`nlp/preprocessing.py`, `nlp/ner.py`, `nlp/severity.py`, and
`nlp/embeddings.py`.
