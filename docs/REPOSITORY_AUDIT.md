# Repository audit

Refresh: 2026-11-09. Canonical repository: `D:\Wahid-ML-Setup\IOT-Predictive-Maintenance-NLP`.

- Branch `main`, HEAD `aabe917b7a123d1f8cc90618afe18b49c3011070`, upstream
  `origin/main`.
- The worktree contained extensive intentional modified and untracked work;
  it was preserved. No reset, clean, checkout, branch switch, commit, push, or
  destructive restore was performed.
- Existing architecture includes FastAPI, React/Vite, Tauri configuration,
  FD001 preprocessing/LSTM artifacts, NLP/fusion adapters, MQTT/Influx local
  integration, assistant storage/grounding, tests, and audit tooling.
- Python 3.11.9 and Node 22/npm 10 are available. Cargo/Rust and Ollama are
  unavailable. Existing dependencies imported successfully; no unrelated
  dependency upgrade was performed.
- The PR13 compatibility patch was absent initially and was applied only to
  `scripts/verify_nlp_fd001_compatibility.py`; it now matches the reference
  branch for that file.
- `artifacts/preprocessing/fd001/` remained unchanged and its handover
  verifier passed.

The final Python suite passed 427 tests, skipped 4, emitted 2 warnings, and
completed 25 subtests. The only recorded warnings are the existing pytest
return-value warning and a scikit-learn serialized-version warning.
