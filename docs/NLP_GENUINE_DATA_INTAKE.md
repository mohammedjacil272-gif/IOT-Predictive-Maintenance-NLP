# Genuine technician-log intake

The nlp.technician_logs module and scripts/validate_genuine_logs.py provide a
fail-closed, dry-run intake boundary for an owner-supplied CSV. The input must
contain stable source IDs, an approved canonical unit and cycle mapping,
timezone-aware event/creation/ingestion/availability timestamps, source
dataset/version, provenance reference, is_synthetic=false, and an approved
status.

Validation preserves empty rows as explicit empty_text records, rejects
duplicate log/source identities and invalid mappings, normalizes text
deterministically, sorts by log_id, and reports multiple records sharing a
unit/cycle. It never infers mappings, drops records, downloads a transformer,
generates embeddings, or writes artifacts.

Example dry run after an owner supplies an approved CSV:

    python scripts/validate_genuine_logs.py path/to/approved_logs.csv --dataset owner-export --dataset-version 2026-01 --reject-empty-text

Successful intake is necessary but not sufficient for fusion. The embedding
run still requires an approved local model/tokenizer revision, mapping and
timestamp/cutoff approvals, immutable artifact hashes, and the genuine fusion
training authorization. The synthetic fixture builder remains explicitly
limited to dataset_type=synthetic_fixture.
