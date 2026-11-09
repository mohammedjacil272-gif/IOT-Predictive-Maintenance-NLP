# NLP pipeline

Status: rule-based NLP implemented and verified; transformer embedding path
optional and unavailable without approved local weights.

The local path preserves original text, normalizes Unicode/whitespace, extracts
rule-based entities, and computes heuristic severity. `nlp.technician_logs` is
the fail-closed intake boundary for approved genuine CSV data: it validates
source IDs, equipment/cycle mapping, approval, provenance, and timezone-aware
timestamps, retains empty rows explicitly, sorts by `log_id`, and reports
multiple logs per unit/cycle.

The transformer path uses local-only DistilBERT loading and never downloads
weights during application startup. No trained NER or severity metrics are
claimed because no labeled genuine technician corpus is available.
