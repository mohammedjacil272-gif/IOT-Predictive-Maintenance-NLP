# Multimodal Fusion — Complete Technical Audit

Audit date: 2026-11-08
Repository: `C:\Users\moham\Desktop\IOT_Integrated`
Scope: read-only code, artifact, documentation, and verification audit. This report is the only new file.

## Executive finding

The repository contains a substantial, fail-closed *fusion interface and validation pipeline*: embedding artifact adapters, identity/key alignment, policy-controlled missing/multiple-log handling, point-in-time and provenance checks, immutable validated sample containers, entity-level split manifests, and a 64-D/768-D fusion network. Synthetic fixtures exercise these contracts.

It does **not** contain a trained or validated multimodal predictive model. No genuine paired sensor/log dataset, approved mapping, genuine fusion checkpoint, fusion training history, validation prediction/metrics, or independent fusion test result is present. `training/train_multimodal.py` is a contract loader/dataset compatibility layer whose `main()` deliberately raises an error saying training is blocked. `scripts/verify_fusion.py` is a synthetic forward-pass smoke test only. The FastAPI fusion route returns `prediction: null` and `not_validated`; the dashboard reports that state.

## Status labels

- **COMPLETE AND VERIFIED**: implementation and the relevant software checks are present and pass.
- **IMPLEMENTED BUT NOT FULLY VERIFIED**: interface/guards exist, but real-data/live/model evidence is missing.
- **PARTIALLY IMPLEMENTED**: a limited component exists without the end-to-end behavior needed.
- **BLOCKED BY EXTERNAL DEPENDENCY**: requires real data, mapping, approval, or model assets not present.
- **NOT IMPLEMENTED**: no functional component in this checkout.

## 1. Current architecture and flow

### Sensor branch

`data/train_FD001.txt` → `preprocessing/cmapss_fd001.py` → canonical C-MAPSS records and frozen preprocessing → 17 selected features and 30-cycle windows → `models/lstm_baseline.py:LSTMBaseline` → RUL output and final-hidden-state sensor embedding → `artifacts/embeddings/fd001_lstm_baseline/sensor_embeddings.npy` with metadata.

The fusion preparation layer in `training/train_multimodal.py` reads existing sensor embeddings and metadata, binds them to approved provenance, and creates sensor records keyed by `(unit_id, label_cycle)`. It expects a 64-dimensional vector, split, window bounds, RUL label, namespace, artifact identity/hash, and timestamps. It does not call the LSTM per fusion sample; it consumes precomputed embeddings.

### Technician-log branch

Log CSV/artifact package → `scripts/build_nlp_artifacts.py` and `nlp/{preprocessing,ner,severity,embeddings}.py` → processed rows, entities, heuristic severity, embedding tensor and metadata → `nlp/fusion_adapter.py` or `training/nlp_fusion_adapter.py` → validated text records keyed by `(unit_id, cycle)`.

The fixture package has five synthetic logs, 16 rule-extracted entities, and a reported tensor shape of `(5,768)`. Its verification report identifies `dataset_type: synthetic_fixture`; severity is heuristic and rule NER is unevaluated against labeled real logs.

### Fusion branch

`training/nlp_fusion_pipeline.py` validates the NLP artifact package, adapts its embeddings and annotations, then calls `training/fusion_dataset.py:build_fusion_samples`. The builder joins by canonical integer `(unit_id, cycle)`, checks namespace, dimensions, split consistency, provenance, and point-in-time policies, and returns validated sensor/text tensors, labels, identity/metadata, and `has_log`. `training/train_multimodal.py:make_dataset` only accepts the strict validated object. A `MultimodalFusionModel` forward call can consume those tensors, but no authorized training path completes the learning/evaluation lifecycle.

## 2. Existing implementation inventory

| Area | Files | Status |
|---|---|---|
| Fusion network | `models/multimodal_fusion.py` | IMPLEMENTED BUT NOT FULLY VERIFIED |
| Keyed alignment, namespace/dimension checks, missing/multiple log policy | `training/fusion_dataset.py` | IMPLEMENTED BUT NOT FULLY VERIFIED |
| Approval/provenance/time validation | `training/fusion_provenance.py` | IMPLEMENTED BUT NOT FULLY VERIFIED |
| Entity-disjoint split manifest and hash | `training/fusion_splits.py` | COMPLETE AND VERIFIED as a utility; genuine split remains unapproved |
| Sample/embedding integrity | `training/fusion_integrity.py`, digest/receipt logic in dataset | COMPLETE AND VERIFIED for covered software cases |
| Text artifact adapters | `nlp/fusion_adapter.py`, `training/nlp_fusion_adapter.py` | IMPLEMENTED BUT NOT FULLY VERIFIED |
| Artifact validation/orchestration | `training/nlp_artifact_validation.py`, `training/nlp_fusion_pipeline.py` | IMPLEMENTED BUT NOT FULLY VERIFIED |
| Training data/config gate | `training/train_multimodal.py` | PARTIALLY IMPLEMENTED; no training run |
| Metrics | `training/fusion_metrics.py` | COMPLETE AND VERIFIED as a metric helper only |
| Model forward smoke test | `scripts/verify_fusion.py` | COMPLETE AND VERIFIED as synthetic shape/forward check |
| Provenance/fixture verifier | `scripts/verify_fusion_provenance.py` | COMPLETE AND VERIFIED as synthetic contract verification |
| FD001 compatibility verifier | `scripts/verify_nlp_fd001_compatibility.py` | PARTIALLY IMPLEMENTED; currently has a verifier-call signature defect (details below) |
| Fusion evaluation wrapper | `scripts/evaluate_multimodal.py` | PARTIALLY IMPLEMENTED; no actual model/checkpoint result |
| API/UI fusion inference | `app/main.py`, `app/static/index.html`, `app/static/app.js` | honest unavailable status; genuine fusion inference NOT IMPLEMENTED |

Relevant tests include `test_fusion_shapes.py`, `test_fusion_alignment.py`, `test_fusion_provenance.py`, `test_fusion_integrity.py`, `test_fusion_adapter.py`, `test_missing_text.py`, `test_nlp_fusion_adapter.py`, `test_nlp_fusion_pipeline.py`, `test_nlp_fusion_integration.py`, and `test_nlp_fusion_end_to_end.py`.

## 3. Sensor branch readiness

**COMPLETE AND VERIFIED** for the existing sensor-only baseline and software contract.

- Canonical raw schema: unit ID, cycle, three operating settings, 21 sensors.
- LSTM input feature order (17): `setting_1`, `setting_2`, `sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`.
- Sequence length: 30 cycles. Baseline split: 80/20 engines, seed 42. Scaler fit scope is training-engine rows only.
- LSTM configuration: 17 inputs, two layers, hidden size 64; the final hidden state is the 64-D representation.
- The local tree has `best_model.pt` and `last_model.pt`; the checkpoint is ignored by Git and is local artifact state. The inference adapter `iot/lstm_inference.py` validates 30 consecutive cycles, feature keys/order, finite values, single unit, and forbidden target fields.
- Baseline data report records train `(14207,30,17)`, validation `(3524,30,17)`, test `(100,30,17)`; sensor embeddings cover train+validation windows.
- The separate official FD001 evaluation report exists under `artifacts/integration_eval_20261108/`; it is sensor-only, not a fusion score.

Fusion sees **the 64-D learned representation**, not the raw 30×17 sequence. `training/train_multimodal.py` consumes the precomputed embedding array with its metadata index and hash. This establishes a software path, not a valid sensor/log pairing.

## 4. NLP, embeddings, and API connectivity

### NLP components

- `nlp/preprocessing.py`: Unicode normalization, whitespace collapse, lowercase normalized text, source preservation.
- `nlp/ner.py`: deterministic phrase/rule NER, with source-text offsets and rule IDs.
- `nlp/severity.py`: lexicon score 0–4; highest unnegated severity word wins.
- `nlp/embeddings.py:TextEmbedder`: optional local-only DistilBERT tokenizer/model; attention-mask mean pooling; 768 dimensions; no network download attempted.
- NLP artifact metadata/report tracks fixture identity, model name/config, rows, entities, severity, and embedding shape. The manifest/provenance utilities are present; fixture records do not become genuine through validation.

### Readiness

- Rule-based clean/NER/severity: **COMPLETE AND VERIFIED** for deterministic software behavior, not accuracy.
- DistilBERT embedding implementation: **IMPLEMENTED BUT NOT FULLY VERIFIED**. The environment audit previously showed `transformers` absent and no local approved model artifacts; runtime diagnostics report unavailable. A checked-in 5×768 fixture tensor is present, but the fixture report describes a frozen embedding package and does not establish current live embedder availability or authentic real-log representations.
- Entity identity/time provenance for genuine records: **BLOCKED BY EXTERNAL DEPENDENCY**. Source system IDs, approved mapping, authoritative event/creation/ingestion/availability times, and immutable real data lineage are not supplied.
- Dashboard log API: **PARTIALLY IMPLEMENTED**. `POST /api/technician-log/analyze` returns rule entities/severity and says embedding `not_generated`. The route does not call the artifact adapter, does not write a fusion package, and does not align with sensor records.
- Therefore, current API rule entities and severity are **not connected as predictive inputs** to the fusion model. Fusion consumes 768-D text embeddings supplied through validated artifact records. Entities/severity are annotations/provenance; they are not embeddings and are not the learned model representation.

## 5. Pairing, alignment, and leakage controls

### Implemented software protections

- Join by `(unit_id, cycle)`, not row order.
- Explicit namespace matching; synthetic and genuine namespaces cannot be mixed.
- Positive integer identity/cycle, unique keys, explicit split, sensor feature dimension 64, text dimension 768, finite tensors.
- A unit must stay in one split; mismatched sensor/text split membership is rejected.
- Missing text policy must be explicit (`zero_vector` or error); multiple-log policy must be explicit (mean sorted by log ID or error). Zero vector and `has_log=False` are returned for missing text.
- Point-in-time logic parses timezone-aware timestamps and compares log availability to prediction cutoff under an explicit policy.
- Genuine records require artifact/source/mapping/model lineage and digest checks against an approved manifest; training loader checks explicit approval, model/schema identity, artifact hashes, source-record embedding lookup, and approved policies.
- Split utility creates deterministic canonical manifests and rejects equipment overlap. Adding a final test composition requires an evidenced approval object.

### Limitations

These are contract checks, not proof the fields are truthful. The fixture IDs and cycles are synthetic. No evidence establishes that an FD001 engine and a technician fixture ID refer to the same physical asset or that FD001 cycle counts share semantics with log timestamps. The current basic alignment key uses cycle identity; the temporal provenance fields/cutoff policy needed for genuine time alignment require authoritative source metadata and approval. The current validation logic cannot manufacture that evidence.

**Genuine paired observations: NOT ESTABLISHED.** Current fixture compatibility scripts intentionally inspect FD001 validation metadata and avoid mapping synthetic fixture logs to it.

### Required data-contract/approval inputs

1. Permissioned genuine technician logs and source dataset/version, source system/record IDs, retention/PII rules.
2. Authoritative equipment IDs and a versioned, evidenced crosswalk to sensor unit IDs.
3. Data dictionary for cycle/window end, log event time, creation time, ingestion time, availability time, timezone, and clock precision.
4. Prediction landmark, cutoff inclusivity, operational horizon, and target definition.
5. Provenance/artifact schema, model/tokenizer immutable revision and checksums.
6. Entity-disjoint train/validation/test assignments and immutable split manifest.
7. Approved missing/multiple-log aggregation and `has_log` model-input policy.
8. Explicit genuine fusion training authorization and separate final held-out evaluation authorization.

## 6. Fusion model

`MultimodalFusionModel` in `models/multimodal_fusion.py` is a feed-forward regression head. Defaults: sensor 64, text 768, hidden 128, head 64, dropout 0.2. It concatenates the two vectors, optionally with one `has_log` feature. Layers: Linear(832 or 833→128), ReLU, Dropout, Linear(128→64), ReLU, Linear(64→1). Constructor requires explicit boolean `use_has_log`. Inputs must be floating tensors of identical dtype/device, shapes `[batch,64]` and `[batch,768]`; the optional flag is `[batch]`. Output is `[batch]`.

The declared target is FD001 uncapped RUL in cycles (`max_cycle(unit)-cycle`) at the sensor window’s end/label cycle, inherited from the sensor baseline. Whether this is the approved deployment prediction target/cutoff/horizon remains pending. There is no sequence-level fusion block; temporal sensor context is already compressed into the LSTM embedding.

Missing text is handled in the data builder as a zero 768-D vector, with `has_log=False`; whether this flag enters the network is a required explicit policy. The model does not include a learned text encoder, cross-attention, gating, uncertainty head, or explanation mechanism. Output is one scalar; there is no calibrated confidence/uncertainty.

## 7. Training, metrics, and artifacts

**Genuine fusion training: NOT IMPLEMENTED / BLOCKED BY EXTERNAL DEPENDENCY.**

Evidence:

- `training/train_multimodal.py` performs extensive gated data/artifact loading and TensorDataset construction. Its executable `main()` intentionally raises: “Multimodal training is intentionally blocked; use the approved contract before adding a training entry point.”
- `scripts/evaluate_multimodal.py` is not evidence of a completed run; the checked-in comparison report lists fusion as pending real NLP artifacts/training.
- No fusion checkpoint is in `artifacts/models`; only FD001 `best_model.pt` and `last_model.pt` are present.
- No fusion training history, validation predictions, fitted fusion model config/checkpoint, independent fusion test report, or reproducibility run manifest exists.
- `artifacts/metrics/fd001_multimodal/comparison_report.json` has only sensor validation metrics (MAE 20.8844, RMSE 31.3649, MSE 983.7574). NLP and fusion metric fields are null/pending. It says evaluation split is validation and official test was not evaluated for that comparison.
- Separate sensor official test evaluation exists, but it cannot be substituted for fusion evaluation.
- `fusion_metrics.py` computes MAE/RMSE/MSE with shape/finite checks. It does not train or establish metric provenance.
- No fusion loss/optimizer/epoch/checkpoint-selection implementation was found in the gated training module. The model’s scalar regression output and data target imply a regression task, but a real training-loss choice/run is not evidenced.

Thus all fusion predictions/metrics from smoke scripts are synthetic interface outputs, not model performance.

## 8. API and dashboard

- `POST /api/fusion/infer` exists but returns `mode=fusion`, `status=not_validated`, `prediction=null`, and a reason. It does not accept/use paired modality payloads, perform alignment, load a fusion checkpoint, or invoke `MultimodalFusionModel`.
- Dashboard shows Fusion readiness as **NOT VALIDATED** and displays the backend reason. It has no paired-input form or fusion submit button, which is appropriate while no validated checkpoint exists.
- Dashboard supports sensor inference and rule-based NLP separately. There is no path that combines both dashboard requests into a fusion sample.
- Overall: honest readiness UI, but genuine fusion inference integration **NOT IMPLEMENTED**.

## 9. Test and verification results

### Fusion and related tests

Command:

```powershell
python -m pytest -p no:cacheprovider -q tests/test_fusion_shapes.py tests/test_fusion_alignment.py tests/test_fusion_provenance.py tests/test_fusion_integrity.py tests/test_fusion_adapter.py tests/test_missing_text.py tests/test_nlp_fusion_adapter.py tests/test_nlp_fusion_pipeline.py tests/test_nlp_fusion_integration.py tests/test_nlp_fusion_end_to_end.py
```

Result: **169 passed in 3.54s**.

### Verification scripts

- `$env:PYTHONPATH='.'; python scripts/verify_fusion.py`: PASS. Synthetic tensors (batch 32, 64-D/768-D), 32 outputs. Script states no optimizer, backward pass, training loop, or artifact write.
- `python scripts/verify_fusion_provenance.py`: PASS for five synthetic fixture logs, five 768-D finite embeddings, 16 entities, alignment, severity preservation, missing text and fixture split checks. Prints “No training or model evaluation performed.”
- `python scripts/verify_nlp_fd001_compatibility.py`: PASS after caller arguments were reconciled with `build_fixture_samples(sensor_records, text_records)`. Internal policies remain fixed to `zero_vector` and `mean_sorted_log_id`; strict dimensional, identity, and split checks remain active. No fixture records were mapped into FD001 validation windows.
- `python scripts/verify_nlp.py --package-dir artifacts`: BLOCKED because the nested fixture files resolve successfully, but its required `manifest.json` is absent. The validator now recognizes `artifacts/data/fd001_nlp_fixture/` and `artifacts/embeddings/fd001_nlp_fixture/`. No manifest or artifact was generated to work around the missing provenance record.
- Compatibility regression tests and full-suite results are recorded in the “Verification compatibility follow-up” section below.
- No training, official test, or live-service evaluation was performed by this audit.

## 10. Exact missing components

1. Permissioned genuine paired logs and sensor observations with documented versions.
2. Approved machine crosswalk and cycle/time semantics.
3. Authoritative log availability and prediction cutoff/landmark policy.
4. Approved target/horizon and fusion comparison protocol.
5. Genuine artifact manifest with source and model/tokenizer revisions/hashes.
6. Approved paired train/validation/test split; evidence that units are disjoint.
7. A real embedding run on approved logs with local/pinned model assets and row lineage.
8. A working, authorized fusion training entry point, loss/optimizer/scheduler/seeds/checkpoint selection, training history and validation-only selection.
9. Sensor-only and NLP-only baselines on the same eligible examples/splits.
10. Independent frozen fusion test evaluation after model/policy freeze and separate authorization.
11. Fusion checkpoint loader, artifact manifest, inference service, and actual paired API contract.
12. Dashboard workflow for valid paired inputs, showing alignment/availability/missing modality and only validated outputs.
13. Supply an authentic, reviewed NLP artifact `manifest.json`; the CLI path compatibility is fixed, but manifest validation correctly remains blocked.
14. Reproducible dependency/model lock and run manifest.

## 11. Recommended implementation order

1. Resolve approvals and data access first: source permission, PII, labels/target, machine mapping, time dictionary, cutoffs, operational horizon, missing/multiple-log and `has_log` policy.
2. Freeze schemas and immutable manifests; independently audit a real paired sample and prove no leakage.
3. Produce and validate real text embeddings with pinned local model/tokenizer revision; retain row/source hashes.
4. Create equipment-disjoint train/validation/test manifest and baseline populations; do not modify frozen FD001 artifacts.
5. Implement a separate authorized fusion trainer using only train data, validation-only selection, complete configuration/history/checkpoints and reproducible seeds.
6. Evaluate sensor-only, NLP-only and fusion models on matched eligible splits; report counts/coverage and compare target/population consistently.
7. Freeze model/policy, obtain separate final-test authorization, then run one independent held-out evaluation; never tune on official FD001 test labels.
8. Add manifest-validated fusion checkpoint loading and a typed API contract that performs approved alignment/cutoff checks.
9. Connect dashboard fusion operation only after readiness checks pass; retain visible modality and provenance explanations.
10. Keep verifier regression coverage; supply the authentic manifest through the normal artifact provenance process, then rerun the CLI and document model card, data card, and limits.

## 12. Summary status

| Capability | Status |
|---|---|
| Sensor baseline and 64-D embedding artifact | COMPLETE AND VERIFIED |
| Rule NLP / fixture artifact plumbing | COMPLETE AND VERIFIED for software contracts only |
| Local transformer embedding implementation | IMPLEMENTED BUT NOT FULLY VERIFIED |
| Identity/key alignment utilities | IMPLEMENTED BUT NOT FULLY VERIFIED |
| Genuine sensor-log pairing | BLOCKED BY EXTERNAL DEPENDENCY |
| Point-in-time/provenance guards | IMPLEMENTED BUT NOT FULLY VERIFIED; genuine evidence absent |
| Fusion model architecture/forward pass | COMPLETE AND VERIFIED for shapes/synthetic execution |
| Fusion training/checkpoint/metrics | NOT IMPLEMENTED |
| Genuine fusion evaluation | BLOCKED BY EXTERNAL DEPENDENCY |
| Fusion API/dashboard inference | NOT IMPLEMENTED; UI reports unavailable honestly |
| Genuine multimodal predictive maintenance | BLOCKED BY EXTERNAL DEPENDENCY |

The original audit was read-only apart from this report. The compatibility follow-up below additionally changed only the two verifier/CLI Python scripts, the artifact path resolver, a new regression test module, and this report. No dataset, preprocessing artifact, checkpoint, generated NLP artifact, or Git history was modified.

## 13. Verification compatibility follow-up — 2026-11-08

### Minimal fixes

- `scripts/verify_nlp_fd001_compatibility.py`: removed unsupported policy keywords from calls to `build_fixture_samples()`. The wrapper continues to select the established `zero_vector` missing-log policy and `mean_sorted_log_id` multiple-log policy. Validation was not weakened.
- `training/nlp_artifact_validation.py`: added lookup candidates for the actual namespaced `data/fd001_nlp_fixture/` and `embeddings/fd001_nlp_fixture/` directories, retaining support for existing flat and split layouts.
- `scripts/verify_nlp.py`: clarified in CLI help that `--package-dir` may point to the artifacts root containing the namespaced fixture package. The default remains `artifacts`.
- `tests/test_verifier_compatibility.py`: regression tests cover missing-log policy preservation and nested fixture resolution with honest blocking when the manifest is absent.

### Verification commands and results

FD001 compatibility verifier:

```powershell
$env:PYTHONPATH='.'; python scripts/verify_nlp_fd001_compatibility.py
```

Result: **PASS (exit 0)**. Five actual fixture embeddings were validated; dimensionality, metadata agreement, integer alignment keys, no fabricated FD001 validation mapping, rejection of 767-D embeddings, inconsistent unit metadata, and fabricated validation split metadata all passed. The script explicitly reports this as a synthetic fixture interface test and performs no training/evaluation.

NLP artifact CLI:

```powershell
$env:PYTHONPATH='.'; python scripts/verify_nlp.py --package-dir artifacts
```

Result: **BLOCKED (exit 1)** with `Missing required artifact: manifest.json`. The nested data and embedding directory layout is now recognized; the only reported blocker is the missing required manifest. No manifest was fabricated and no artifact was moved, overwritten, or regenerated.

Fusion/NLP regression and component tests:

```powershell
$env:PYTHONPATH='.'; python -m pytest -p no:cacheprovider -q tests/test_fusion_shapes.py tests/test_fusion_alignment.py tests/test_fusion_provenance.py tests/test_fusion_integrity.py tests/test_fusion_adapter.py tests/test_missing_text.py tests/test_nlp_fusion_adapter.py tests/test_nlp_fusion_pipeline.py tests/test_nlp_fusion_integration.py tests/test_nlp_fusion_end_to_end.py tests/test_nlp_artifact_validation.py tests/test_nlp_artifact_manifest.py tests/test_nlp_embeddings_diagnostics.py tests/test_ner_measurements.py tests/test_verifier_compatibility.py
```

Result: **224 passed, 2 skipped**.

Full test suite:

```powershell
python -m pytest -p no:cacheprovider -q
```

Result: **372 passed, 3 skipped, 1 warning, 25 subtests passed**. The warning is pre-existing `PytestReturnNotNoneWarning` from `tests/test_influx_ingestion.py::test_config`, which returns an `InfluxIngestionConfig` object. No test failures.

The authentic manifest remains necessary before the artifact CLI can report PASS. All fusion training, genuine paired-data validation, and model-performance limitations from this audit remain unchanged.
