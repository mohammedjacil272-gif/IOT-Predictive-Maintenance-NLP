# Remaining blockers

1. Obtain and approve real technician logs, labels, source IDs, machine crosswalk, timestamps, and PII policy.
2. Approve fusion target, cutoff/landmark semantics, missing/multiple-log policy, and training authorization.
3. Provide pinned local transformer/tokenizer assets if text embeddings are required offline.
4. Decide how ignored FD001 checkpoints are distributed for clean-clone reproducibility.
5. Run live MQTT/Influx tests only in an explicitly configured local environment.
6. Define authentication, retention, and deployment requirements before exposing the service beyond localhost.
7. Browser click-through and console validation were not automated in this run.
8. A permissioned technician-log export, approved machine crosswalk, shared temporal semantics, target/cutoff, and entity-disjoint paired package are still required.
9. Browser click-through remains manual; no browser automation was installed.
10. Assistant speech/TTS providers, production authentication/authorization, and deployment hardening remain unimplemented.
11. Existing `.assistant.sqlite3` and `.local_history.json` were not migrated automatically; run the tested migration deliberately after reviewing backup location and retention policy.
12. Native Windows build is blocked by missing Rust/Cargo, Tauri CLI, PyInstaller and completed npm dependency installation.

Synthetic fixture verification and successful forward passes do not remove these blockers.
