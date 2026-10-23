# Reproducibility and Dependencies

**Audit reference:** `1ffc5be56b697a6fe3c97df0c32c1b5ed0f5bf3f`. This is a plan, not a lockfile. Keep observed environment versions separate from declared support.

## Findings at audit time

- No tracked dependency manifest/lockfile was found among common names (`pyproject.toml`, `requirements*.txt`, Conda environment file, Poetry/uv lock, Pipfile, `setup.py`, `setup.cfg`).
- A previous user-observed Windows environment reported Python 3.11.9, pytest 9.1.1, torch 2.14.1, transformers 5.19.0, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, and joblib 1.6.0. That observation came from the earlier Wahid worktree and is not, by itself, the supported matrix for this audit SHA.
- A separate submitted handoff claimed Python 3.13.7, PyTorch 2.14.1, NumPy 2.5.4, pandas 3.0.6, pytest 9.1.1 and scikit-learn 1.9.1. This was not independently tied to audit SHA `1ffc5be`; treat it as an unverified separate environment claim.
- `paho-mqtt` and `influxdb-client` were not installed in the previously inspected user Python environment. Their imports are lazy in the IoT paths; mock tests can run without live clients/services being exercised.
- Supported Python version/platform policy: **PENDING MAINTAINER APPROVAL**.

## Dependency inventory by capability

Based on tracked imports:

- Core fusion/tests: Python standard library, `torch`, `pytest`.
- Sensor preprocessing/baseline: `numpy`, `pandas`, `scikit-learn`, `joblib`, `torch`.
- NLP embedding generation: `transformers` and its runtime dependencies plus model/tokenizer assets.
- Live MQTT path: `paho-mqtt`.
- Live InfluxDB path: `influxdb-client`.

This is an import inventory, not a fully resolved lock. Binary wheels for PyTorch and its dependencies vary by OS, Python ABI, and CPU/CUDA runtime.

## Proposed reproducible manifest strategy — PENDING APPROVAL

1. Maintainer declares supported Python versions and target platforms.
2. Review dependencies into core/test, NLP-generation, and optional IoT groups if the chosen tooling supports it.
3. Resolve direct and transitive dependencies in a clean environment, record package versions and hashes where practical, and commit an approved manifest plus lock/constraints file.
4. Run CI on the declared supported runtime(s), installing only from the committed declaration.
5. Do not treat `pip freeze` from a single unreviewed environment as a supported lock.
6. Do not install packages globally.

Example isolated environment commands **after Python support is approved**:

```powershell
py -3.X -m venv .venv
.\.venv\Scripts\python.exe -m pip --version
# Install only from the future approved manifest/lock.
```

`3.X` is a placeholder, not a command to run before the supported version is chosen.

## Hugging Face text-model reproducibility — BLOCKED

At the audit SHA, `nlp/embeddings.py` declares:

- `MODEL_NAME = "distilbert-base-uncased"`.
- `MAX_LENGTH = 128`.
- `EMBEDDING_DIM = 768`.
- `AutoTokenizer.from_pretrained(MODEL_NAME)` and `AutoModel.from_pretrained(MODEL_NAME)` without an explicit immutable `revision`.
- Model is placed in eval mode and parameters are frozen.
- Pooling is a mean of `last_hidden_state` weighted by `attention_mask`, then returned as CPU float output.

The model/tokenizer revision and model-file hashes are missing from loader/report evidence. Do not infer which exact Hub revision produced an old tensor from the model name alone.

Required follow-up with the NLP owner:

1. Approve an immutable model repository commit/revision for both model and tokenizer. Add `revision=` to the loader only in a separately reviewed code change.
2. Record model repository, immutable revision, tokenizer revision, config, max length, pooling, text preprocessing revision and relevant library versions in artifact metadata.
3. After the approved assets are obtained, calculate SHA-256 hashes for model/tokenizer files and the output tensor; retain a manifest with the artifact build ID.
4. Document the Hugging Face cache path/default behavior, whether first-use network downloads are permitted, the approved `HF_HOME` override if used, and how to run with a populated cache/offline mode.
5. Do not include secrets or weights in the documentation bundle. Do not regenerate fixture artifacts unless explicitly authorized; compare outputs before replacing any tracked fixture.

A digest establishes byte identity against a known digest, not source authenticity or data truth.

## Frozen baseline checkpoint availability — BLOCKED

At audit SHA, neither `best_model.pt` nor `last_model.pt` appeared in the tracked asset inventory or the on-disk search of the audit worktree. `scripts/verify_lstm_baseline.py` lists both as required trained artifacts, and `iot/lstm_inference.py` points to `best_model.pt` by default.

Do not run the baseline verifier blindly during this read-only audit: its data verification path may call baseline data preparation when artifacts are missing. Do not regenerate data, retrain, alter scaler/preprocessing, or synthesize replacement checkpoints. The baseline owner must locate authorized, matching checkpoint artifacts and establish hashes/config/metric lineage before claiming baseline inference can be reproduced.

## Safe non-training validation commands

After the repository owner approves a runtime and a clean worktree is available, capture commit SHA, interpreter path/version, dependency environment, command, exit code and output for:

```powershell
python --version
python -m pytest tests/test_fusion_alignment.py -q
python scripts/verify_fusion_provenance.py
python scripts/verify_nlp_fd001_compatibility.py
python -m pytest -q
git diff --check
git status --short --branch
```

Do not run fusion training/evaluation or official FD001 test evaluation as part of readiness work. Bind all test counts to exact commits and actual output.
