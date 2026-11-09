# Rafaz NLP remaining requirements

## Current evidence

The local NLP pipeline provides normalization, rule-based entities, heuristic severity, provenance fields, artifact validation, and a five-record synthetic fixture. The fixture's 768-D tensor is interface evidence only; it is not genuine technician narrative data and must not enter genuine fusion training.

The supplied Rafaz ZIP was inspected read-only. Its documents support the same distinction between interface tests and production readiness. No ZIP content establishes project authorization, raw-file acquisition, or genuine narratives.

## Azure PdM audit

Rafaz's audit identifies filenames at pinned commit `e836f4e866028db7fd7d1a66b88f79664a1f4eac` in `microsoft/sqlworkshops/SQLServerAndAzureMachineLearning/ML Services for SQL Server/data/`: `PdM_telemetry.csv`, `PdM_machines.csv`, `PdM_errors.csv`, `PdM_maint.csv`, and `PdM_failures.csv`. This is filename verification only. The data is a simulated benchmark, not genuine technician narratives. No local project acquisition manifest or authorized five-file copy was found, so no download/copy/hash claim is made.

## Required evidence from Rafaz

- Dataset owner, licence, project-specific permission, retention and redistribution decision.
- Authorized acquisition manifest with exact source commit, file sizes and SHA-256 hashes, created only after actual authorized acquisition.
- Genuine technician narrative source, stable `log_id`, source timestamps, availability timestamps, PII/retention approval, and provenance lineage.
- Approved mapping from source equipment identifiers to canonical `unit_id`; no inferred mapping from synthetic fixtures.
- Approved timestamp/cycle conversion or a defensible common alignment rule, including cutoff and late-arrival policy.
- Exact DistilBERT model and tokenizer identifiers, immutable revisions, tokenization/max length, pooling, dtype, output shape, and model/tokenizer file hashes.
- NLP run report with schema, duplicate/empty/negation/conflict handling, finite-value checks, row-to-`log_id` mapping, and fixture isolation.

Until these are supplied, NLP is ready for software/interface testing only and genuine fusion remains blocked.
