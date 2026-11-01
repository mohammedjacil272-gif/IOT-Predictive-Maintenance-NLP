# Reproducibility and Dependencies

## 1. Observed Environment

The local virtual environment reported Python 3.13.7 during this review.

| Package | Observed version | Use in this repository |
|---|---|---|
| PyTorch (`torch`) | 2.14.1 | Fusion model, tensor adapter, NLP embeddings, and tests |
| NumPy | 2.5.4 | Metrics and preprocessing |
| pandas | 3.0.6 | Sensor and artifact metadata handling |
| pytest | 9.1.1 | Test runner |
| scikit-learn | 1.9.1 | Baseline scaler and preprocessing |
| joblib | 1.6.0 | Scaler artifact serialization |

These versions are observations from the local worktree, not officially supported or locked dependency versions.

The test suite uses Python, pytest, PyTorch, NumPy, and pandas. Sensor preprocessing and baseline execution additionally use scikit-learn and joblib. The NLP artifact-generation pipeline also depends on Transformers and model assets. The checked-in fixture compatibility verifiers use saved embeddings and do not generate new embeddings.

## 2. Dependency Manifest Gap

No `requirements.txt`, `pyproject.toml`, `Pipfile`, Conda environment file, lockfile, or equivalent dependency manifest was found in the inspected repository root or tracked file inventory.

The local `.venv` is environment state and is not a reproducible dependency declaration. No packages were installed or upgraded for this documentation review.

Recommended follow-up:

1. Project owners should agree on supported Python and package versions.
2. Add a minimal test/runtime dependency manifest and a lockfile or another version-pinning mechanism.
3. Validate the declared dependencies in a clean environment.

Do not infer officially supported versions from the currently installed virtual environment.

## 3. Current Environment Workflow

The local environment can be recreated at the Python-version level using PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Package installation is intentionally omitted until the project owners approve the dependency manifest and package versions.

From the project root, the current validation commands are:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\verify_fusion_provenance.py
.\.venv\Scripts\python.exe scripts\verify_nlp_fd001_compatibility.py
.\.venv\Scripts\python.exe scripts\verify_lstm_baseline.py
```

The provenance and NLP–FD001 compatibility commands validate fixture interfaces, metadata, provenance, and invalid-mapping rejection. They do not establish genuine sensor/log pairing, train the fusion model, or demonstrate predictive improvement.

The baseline verifier is expected to remain blocked until the original required checkpoint files are recovered and verified.

## 4. Recorded Verification Results

### 4.1 Exact source state

The latest recorded verification was run from the project root on 2026-10-29.

- Commit: `19e6de42679246eba36824c1e422d913f923e3ae`
- Branch: `feature/wahid-multimodal-fusion`
- Remote branch: `origin/feature/wahid-multimodal-fusion`
- Git working tree: clean at the time of verification
- Branch synchronization: synced with origin at the time of verification
- Python: 3.13.7, as previously observed in the local environment

The commit and clean/synchronized status were verified in PowerShell before the validation commands were run.

### 4.2 Test suite

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Result:

**67 passed in 22.30 seconds.**

The test suite completed without failed or skipped tests.

The cache provider was disabled because the environment has encountered permission restrictions involving pytest cache access.

The root-discovery command `python -m pytest -q` previously failed during collection because an ignored scratch directory was inaccessible to the environment. The scoped `tests` command above completed successfully.

### 4.3 Fusion provenance verification

Command:

```powershell
.\.venv\Scripts\python.exe scripts\verify_fusion_provenance.py
```

Result: **PASS**

Reported checks included:

- Five synthetic fixture logs loaded.
- Five 768-dimensional text embeddings with finite values.
- Provenance validated for 16 entities.
- Canonical 64-dimensional sensor and 768-dimensional text interface verified.
- Five per-log severity records preserved.
- Missing-text handling verified.
- Fixture split isolation preserved.

The script explicitly reported that no training, model evaluation, or official FD001 test evaluation was performed.

### 4.4 NLP–FD001 compatibility verification

Command:

```powershell
.\.venv\Scripts\python.exe scripts\verify_nlp_fd001_compatibility.py
```

Result: **PASS**

Reported checks included:

- Five NLP embeddings loaded, each 768-dimensional.
- NLP metadata, processed logs, and severity records agreed.
- Alignment keys were integer `(unit_id, cycle)` pairs.
- Fixture split metadata was preserved.
- FD001 validation metadata was inspected without fabricating a mapping between fixture records and validation windows.
- Sensor and text embedding dimensions were verified.
- Invalid embedding dimensions and inconsistent equipment metadata were rejected.
- Fabricated FD001 validation split mapping was rejected.

**Limitation:** These are synthetic fixture interface checks. They do not establish genuine sensor/technician-log alignment or real-world predictive performance.

### 4.5 Frozen baseline verification

Command:

```powershell
.\.venv\Scripts\python.exe scripts\verify_lstm_baseline.py
```

Result: **BLOCKED**

The verifier reported that these required original checkpoint files were missing:

- `best_model.pt`
- `last_model.pt`

The baseline directory contained the configuration files:

- `model_config.json`
- `split_config.json`
- `training_config.json`

The verifier is read-only and explicitly refuses to regenerate the missing protected files.

Additional inspection found that:

- Neither checkpoint was present in the inspected local repository.
- Git did not track either checkpoint filename.
- The repository's `.gitignore` excludes `*.pt`.

The baseline artifact owner must provide the original checkpoints or confirm their approved external storage location and provenance. Checksums should be recorded if available.

**Do not retrain the frozen baseline, create dummy checkpoint files, or modify protected baseline artifacts to satisfy verification.**

## 5. Current Project Status and Restrictions

The latest recorded validation state is:

| Check | Status |
|---|---|
| Test suite | PASS — 67 tests |
| Fusion provenance fixture verification | PASS |
| NLP–FD001 compatibility fixture verification | PASS |
| Baseline checkpoint verification | BLOCKED |
| Genuine sensor/technician-log pairing | NOT VERIFIED |
| Genuine fusion training | NOT PERFORMED |
| Fusion predictive-performance evaluation | NOT PERFORMED |
| Official FD001 test evaluation for fusion | NOT PERFORMED |
| Frozen sensor-only baseline modified | NO |

The implementation checks passing does not prove that fusion improves predictive performance.

No fusion training or performance evaluation should begin until the required data, authoritative metadata, policy decisions, approvals, and applicable verification gates have been satisfied.

## 6. Outstanding Dependencies and Ownership

The following items remain unresolved. The project coordinator must confirm the responsible owner and deadline for each item. No owner assignment or deadline is assumed until confirmed.

| Outstanding dependency | Responsible owner | Deadline | Required deliverable |
|---|---|---|---|
| Original baseline checkpoints | Baseline artifact owner | To be confirmed | Original `best_model.pt` and `last_model.pt`, approved storage location, provenance, and checksums if available |
| Authoritative eligible-equipment manifest | Sensor/data owner | To be confirmed | Versioned manifest, schema, identifier field, provenance, and approved train/validation/test equipment assignments |
| Genuine paired datasets | Sensor and technician-log data owners | To be confirmed | Approved sensor dataset, technician maintenance logs, and source provenance |
| Equipment crosswalk | Sensor and technician-log data owners | To be confirmed | Verified mapping between equipment identifiers across both sources |
| Cycle definitions | Sensor and technician-log data owners | To be confirmed | Authoritative definitions of `cycle` in each source and a verified alignment rule |
| Log availability timestamps | Technician-log data owner | To be confirmed | Trustworthy `available_at` semantics and supporting evidence showing when each log became available to the prediction system |
| Data and evaluation contract | Authorized project, data, and evaluation approvers | To be confirmed | Approved target, prediction cutoff, operational horizon, split protocol, missing-log policy, multiple-log aggregation, and `has_log` treatment |
| Fusion training authorization | Project lead and authorized data/evaluation owners | To be confirmed | Explicit, evidence-referenced authorization for the approved dataset, configuration, and training protocol |
| Final held-out evaluation authorization | Project lead and evaluation lead | To be confirmed | Separate authorization, frozen selected model/configuration, held-out evaluation protocol, and reporting plan |
| Dependency manifest and pinned environment | Project engineering owners | To be confirmed | Approved dependency manifest and successful clean-environment validation |
| Live IoT validation, if required by project scope | IoT owner | To be confirmed | Approved device/test setup, broker and database access, test procedure, and recorded results |

The project coordinator should assign owners and deadlines and track the dependencies in parallel. Until the relevant inputs and evidence are received, the corresponding items remain blocked or pending.

## 7. Data Contract and Approval Requirements

Before genuine fusion training can be authorized, the project must have documented and approved decisions for:

1. The prediction target and its definition.
2. The prediction cutoff and information permitted at that time.
3. The operational prediction horizon.
4. Equipment eligibility and permitted train, validation, and test assignments.
5. Equipment-ID crosswalk and sensor/log cycle alignment.
6. Point-in-time availability semantics for technician logs.
7. Handling of windows without a matching log.
8. Aggregation and provenance preservation for multiple logs at the same equipment/cycle.
9. The meaning and use of the `has_log` indicator.
10. The approved evaluation protocol and held-out data protections.

Required approvals must identify the decision owner and include evidence references. A structurally valid approval record does not, by itself, authenticate the authority of its issuer or independently prove the underlying evidence.

Training authorization and final held-out evaluation authorization are separate gates. Neither should be inferred from a configuration default or from successful fixture tests.

## 8. Future Genuine-Run Reproducibility Record

The training harness has validation and metadata-recording mechanisms for a future authorized genuine-data run. These mechanisms do not mean that a genuine run is currently authorized or that all external dependencies are available.

A future genuine run requires, at minimum:

- A clean committed source tree and exact source commit.
- The legitimate sensor baseline checkpoint and its provenance.
- Authoritative equipment and source-artifact metadata.
- Genuine paired sensor and technician-log data.
- Verified equipment and cycle alignment.
- Trustworthy point-in-time availability timestamps.
- An approved preprocessing version.
- NLP model and tokenizer identifiers and revisions.
- An explicitly configured operational horizon.
- Approved data, split, missing-log, aggregation, and evaluation policies.
- Evidence-referenced training authorization.
- Passing validation and baseline-protection checks.

The resulting run record should include the exact command, test results, Python and dependency versions, model and training configuration, random seeds, consumed-input hashes, canonical split identity and hash, checkpoint lineage, and available model/tokenizer checksums. Missing checksums must be represented as unavailable rather than fabricated.

Final held-out evaluation requires its own explicit authorization and approved protocol.

## 9. Exit Criteria and Next Actions

Proceed to genuine-data validation only after the genuine datasets, authoritative eligible-equipment manifest, verified equipment crosswalk, cycle definitions, availability-timestamp evidence, and required data-contract decisions have been supplied and verified.

Proceed to fusion training only when the original baseline artifacts have been recovered and verified, the required training authorization is recorded, and all applicable data and validation gates pass.

Proceed to final held-out evaluation only after the selected model/configuration and evaluation protocol are frozen and separate evaluation authorization is recorded.

Until then:

- Preserve the current implementation and frozen sensor-only baseline.
- Keep baseline verification marked **BLOCKED** until the original checkpoint artifacts are recovered and verified.
- Continue documentation and safe non-training checks.
- Do not fabricate mappings, timestamps, manifests, provenance, or approval evidence.
- Do not claim that fusion improves predictive performance.
- Do not perform unauthorized fusion training or final held-out evaluation.

The next immediate action is for the project coordinator to assign owners and deadlines and follow up with the responsible artifact, sensor, dataset, technician-log, and approval owners in parallel.