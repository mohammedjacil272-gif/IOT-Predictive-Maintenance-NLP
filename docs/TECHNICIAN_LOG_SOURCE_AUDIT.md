# Technician-log source audit

`tests/fixtures/technician_logs.csv` and its derived artifacts are **SYNTHETIC**; their report explicitly says so. Azure PdM files under `experiments/azure_pdm/tests/fixtures` are fixtures, not approved project logs. No genuine CSV/JSON export, MongoDB connector, consent record, or source approval was found. The application accepts user-submitted text for local NLP-only analysis, but that is not an approved training source. The team must provide permissioned logs, source IDs, provenance, retention/PII approval, and timestamp semantics.
