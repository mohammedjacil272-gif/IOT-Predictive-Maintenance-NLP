# Runtime implementation audit

The live worktree contains the FD001 sensor baseline, rule-based NLP, offline-only embedding adapter, guarded fusion dataset/model code, and local FastAPI/static dashboard. Fusion alignment and provenance code is exercised by synthetic contract tests, but no genuine paired data or fusion checkpoint is present. The embedding service is not connected to a live model because `transformers` and local DistilBERT weights are absent. Existing user changes and frozen artifacts were preserved.
