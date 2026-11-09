# Dataset selection decision

Decision date: 2026-11-09.

No external dataset is accepted for genuine MaintainIQ multimodal training in
this worktree. The local FD001 files are a sensor-only benchmark and the NLP
records under `artifacts/` are explicitly synthetic fixtures. They support
contract, alignment, provenance, and smoke tests only.

The EDP wind-farm release remains the preferred acquisition candidate for
sensor/event research, pending lawful download and inspection of actual files.
It is not selected yet because the documented raw directory is absent. The
Swedish paper-machine signal/annotation dataset is the strongest alternative
research lead, but license, schema, language, timestamp cutoff, and acquisition
evidence still require review. Azure Predictive Maintenance is useful for
structured event adapter tests, not genuine technician free-text fusion.

Before selection, record publisher/license/version, raw filenames and SHA-256
hashes, equipment IDs, timestamp semantics, text authenticity/language, join
keys, missingness, duplicate policy, and leakage-safe chronological splits.
