# Data ingestion and validation

Status: implemented for the existing Azure structured-event adapter and the
generic namespace/hash contract; external raw data acquisition remains pending.

`experiments/azure_pdm/schema.py` is the dataset-specific adapter. It validates
required columns, finite numeric sensors, positive machine IDs, explicit
timezone policy, duplicate keys, source-record identity, and event timestamps.
`preprocessing/dataset_registry.py` supplies reusable `DatasetDescriptor`, raw
file SHA-256 inventory, namespace enforcement, and duplicate-key validation.

Rules:

- Raw files are read-only and are never silently rewritten or dropped.
- Every normalized record carries a dataset namespace and source identity.
- `fd001_baseline`, external benchmarks, genuine data, and synthetic fixtures
  must remain separate namespaces.
- Missing timestamps are rejected unless the source contract explicitly permits
  them; no cycle number is converted into a real event timestamp.
- A classification label is metadata, not proof of authenticity.
- Normalized data and manifests belong in versioned experiment directories, not
  in the frozen FD001 artifact paths.

Validation commands:

```powershell
python -m pytest -q tests/test_dataset_registry.py experiments/azure_pdm/tests/test_phase4_foundation.py
python experiments/azure_pdm/validate_acquisition.py --help
```
