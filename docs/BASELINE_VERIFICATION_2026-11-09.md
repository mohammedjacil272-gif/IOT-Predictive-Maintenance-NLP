# Baseline verification — 2026-11-09

Repository: `C:\Users\moham\Desktop\IOT_Integrated`
Branch: `integration/jacil-full-project`
HEAD: `a05509320804f63d9f650c4b1d36b61c7c9e7bca`

Existing tracked and untracked user work was preserved. No branch switch, commit, push, reset, clean, stash, amend, rebase, sibling-worktree edit, database migration, model download, or protected-artifact modification was performed. `docs.zip` was not present in the repository subtree search.

Final local checks: 397 passed, 3 skipped, 1 warning, 25 subtests; frontend build passed; frontend high-severity production audit reported 0 vulnerabilities; Rust format/check passed; Python compilation passed; `git diff --check` passed. Installed Ollama models were enumerated without download and `qwen2.5:3b` returned `READY` through the gateway. `transformers`/DistilBERT weights were unavailable. MQTT/InfluxDB live services were not authorized or tested.

Protected FD001 SHA-256 values matched before/after implementation:

```text
388d6e45f6d9692aee1d8cc9d8de021c4fab4eb0ce8287445eb24973cc2db013  artifacts/models/fd001_lstm_baseline/best_model.pt
01bd4e6b114af77ad2eb3e4dc6088389b8522b6b0f1a1b5e7ba704857a3d5615  artifacts/models/fd001_lstm_baseline/last_model.pt
aa63e09450e2b1f2c0a008132de7b8c7166b0f07d12bab7623e08edf107acdc9  artifacts/data/fd001_lstm_baseline/X_train.npy
b2cc889d6d68de4185e59c53143cdc2be81d429f6f45037439da8126df1a820b  artifacts/data/fd001_lstm_baseline/X_val.npy
833fe9c8a4661f5ad17ccdf40955329168175BF5637918897A0B1D2F4A0E0DC8  artifacts/data/fd001_lstm_baseline/X_test.npy
b24d359d513679387b429563c10a8b5f4799a0d84d49caa595ad63efaae6ab82  artifacts/embeddings/fd001_lstm_baseline/sensor_embeddings.npy
```

The Azure-named CSVs under `experiments/azure_pdm/tests/fixtures/` are local synthetic fixtures, not genuine technician narratives, not proof of acquisition from the pinned Microsoft repository, and not authorized fusion-training data.
