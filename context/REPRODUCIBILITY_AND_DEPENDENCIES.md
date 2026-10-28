# Reproducibility and Dependencies

## Observed environment

The repository's active virtual environment currently reports Python 3.13.7.
The installed versions observed during this review are:

| Package | Observed version | Use in this repository |
| --- | --- | --- |
| PyTorch (`torch`) | 2.14.1 | Fusion model, tensor adapter, NLP embeddings, and tests. |
| NumPy | 2.5.4 | Metrics and preprocessing. |
| pandas | 3.0.6 | Sensor and artifact metadata handling. |
| pytest | 9.1.1 | Test runner. |
| scikit-learn | 1.9.1 | Baseline scaler and preprocessing. |
| joblib | 1.6.0 | Scaler artifact serialization. |

These are observations from this worktree, not declared supported or locked
versions. The current test suite uses Python, pytest, PyTorch, NumPy, and
pandas. Sensor preprocessing and baseline execution additionally use
scikit-learn and joblib. Running the NLP artifact-generation pipeline also
depends on Transformers and model assets, but the two checked-in fixture
compatibility verifiers consume saved embeddings and do not generate new ones.

## Manifest gap

No `requirements.txt`, `pyproject.toml`, `Pipfile`, Conda environment file,
lockfile, or equivalent dependency manifest was found in the repository root
or tracked file inventory. The `.venv` is local environment state and is not a
reproducible dependency declaration. No packages were installed or upgraded
for this documentation review.

Recommended minimal follow-up: have the project owners choose supported Python
and package versions, add a minimal test/runtime manifest (and lock or other
version-pinning mechanism), then validate a clean environment from that
declaration. Do not infer supported versions from this one installed venv.

## Current environment workflow

The existing local environment can be recreated at the Python-version level
with PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Package installation is intentionally omitted until the dependency manifest
and approved versions exist. In the current environment, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\verify_fusion_provenance.py
.\.venv\Scripts\python.exe scripts\verify_nlp_fd001_compatibility.py
```

The two verifier commands validate the checked-in synthetic fixture and
interface/provenance behavior. They do not train fusion, evaluate model
performance, or establish real-world pairing.

## Recorded verification for this worktree

On 2026-10-28, the full suite was run with the command below from the project
root using `.venv\Scripts\python.exe`, Python 3.13.7, and pytest 9.1.1:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Result: **67 passed** in 8.22 seconds. The suite was tested from fetched base
commit c19e80e86e62a26d4127dcd6f60bc562f79ea4ba with the current uncommitted
worktree changes. The exact tested files are identified by this commit plus
the accompanying final diff/status report, not by the commit alone. Pytest's
cache provider was disabled because this environment denies access to
.pytest_cache.

The requested root-discovery command `python -m pytest -q` was also attempted
and failed during collection because an ignored scratch
directory `tmp6hhbr97a` is inaccessible to this environment. It also emitted
cache-permission warnings. Running discovery from `tests/` avoids scanning
that inaccessible directory and executes all repository tests successfully;
no test files were skipped.

The following fixture-only checks also passed from the same interpreter:

```powershell
\.venv\Scripts\python.exe scripts\verify_fusion_provenance.py
\.venv\Scripts\python.exe scripts\verify_nlp_fd001_compatibility.py
\.venv\Scripts\python.exe -m scripts.verify_fusion
```

No fusion training, predictive performance evaluation, or official FD001 test
evaluation was run.

Earlier reports of 21 and 23 tests have no recorded commit, interpreter,
environment, or exact command, so they cannot be reconciled to a source state.
The 67-test result above supersedes the prior uncommitted 50- and 51-test
reports; the 21- and 23-test reports remain unverified.

## Future genuine-run record

**IMPLEMENTED / GATED:** a genuine run requires a clean committed source tree,
an owner-supplied sensor checkpoint, an explicit preprocessing version, NLP
model and tokenizer names/revisions, an explicitly configured operational
horizon, and a passing verification report matching the exact commit, Python
version, and interpreter. The report must include the exact test command/count
and passing provenance, NLP/FD001 compatibility, and baseline-protection
checks. The resulting run metadata hashes consumed inputs and artifacts,
records the canonical split identity/hash, complete model/training settings,
random seeds, package versions, checkpoint lineage, and available model /
tokenizer checksums. Missing checksums are stored as null; they are not
fabricated.

**BLOCKED:** genuine training remains unavailable until authorized data,
timestamp/mapping/policy approvals, authoritative artifact metadata, the
legitimate sensor checkpoint, and the matching verification report are
provided. The missing `best_model.pt` is not regenerated to satisfy this gate.
