# Wahid — Multimodal Fusion Model & Evaluation Harness

## Project context and completed work
The project predicts RUL from sensor time-series and investigates whether technician maintenance text improves prediction. The active dataset is NASA C-MAPSS FD001. The verified sensor-only LSTM baseline is frozen: 30-cycle windows, 17 features, 80/20 engine split (seed 42), train-only scaler, 64-D sensor embeddings. Best validation metrics: MAE 20.884394, RMSE 31.364907, MSE 983.757406. Official test evaluation has NOT been performed.

## Git setup
```powershell
git clone https://github.com/mohammedjacil272-gif/IOT-Predictive-Maintenance-NLP.git
cd IOT-Predictive-Maintenance-NLP
git switch -c feature/wahid-multimodal-fusion
```

## Your independent workstream
Build fusion and its evaluation harness using mock text embeddings for tests, so you do not need to wait for Rafaz's final embeddings.

## Interface contract
- Sensor embeddings: float array/tensor `[batch, 64]`, traceable to sensor-window metadata.
- Text embeddings: float array/tensor `[batch, 768]`, traceable to `log_id`, `unit_id`, `cycle`, and `split`.
- Canonical key: `(unit_id, cycle)`. `cycle` is the sensor window's `label_cycle` (the window-end cycle). Records are joined by this key, never by array position.
- Shared cycle semantics and mapping between log equipment IDs and sensor equipment IDs are **PENDING validation on genuine paired data**. No mapping is inferred or fabricated here.
- Sensor target: consume the baseline-provided `label_rul` unchanged. Its frozen definition is uncapped `max_cycle(unit) - cycle`; fusion must not recalculate or cap it.
- Output: one unconstrained scalar RUL prediction per joined sample.
- Equipment must belong to one split only. Sensor/text split values must agree; the frozen baseline's train/validation engine assignment is authoritative for real fusion inputs.
- Preserve `log_id`, `unit_id`, `cycle`, and `split`, plus existing entity and severity provenance. Validate supplied provenance instead of silently accepting mismatches.
- Missing text currently yields a zero 768-D vector and `has_log=False`; multiple logs currently sort by `log_id` and use a mean embedding. These are **proposed production policies pending team approval**.
- The adapter now requires callers to explicitly select missing- and multiple-log policies; it does not silently select a production default. Model construction also requires explicit `use_has_log=True/False`. `has_log` is excluded unless selected, and the production choice remains **PENDING approval**.
- Genuine records fail closed unless the approved register supplies dataset, mapping, cycle, timestamp, cutoff/horizon, split/evaluation, embedding-provenance, and log-policy approvals. Genuine log records must preserve distinct `event_time`, `record_creation_time`, `ingestion_time`, and `availability_time` values plus source-system identifiers. Timestamp timezone, cutoff, and late-arrival semantics are not inferred.
- Do not invent timestamps or availability information. The present technician-log pipeline provides no creation/availability timestamps, so point-in-time availability is **PENDING**. Do not claim that a log was available by a sensor window's prediction time without genuine timestamped evidence.
- Prediction cutoff and operational RUL horizon are **PENDING approval**.

## Fixture and evidence boundary

The existing five-record technician-log fixture is synthetic interface/provenance validation only. Its `fixture` split and synthetic sensor vectors must remain identifiable as synthetic; it must never be described as paired FD001 data or real-world fusion performance. The compatibility verifier may inspect real baseline metadata read-only, but must not map fixture rows into baseline train, validation, or test splits. No genuine paired equipment mapping, timestamp, or train/validation/test pairing is currently established.

## Tasks
1. Implement an initial feature-concatenation model: concatenate 64-D sensor and 768-D text vectors, then an MLP regression head. Keep config explicit.
2. Implement metadata-based joining of sensor embeddings, text embeddings, windows and RUL targets.
3. Define deterministic aggregation for multiple logs per window (e.g. mean text embedding) and missing-log handling.
4. Reuse the frozen 80/20 engine split. Do not randomly split windows or fit transforms on validation engines.
5. Train with a regression loss; report MAE, RMSE and MSE. Checkpoint by validation RMSE and use early stopping.
6. Add comparison-report format for sensor-only vs NLP-only vs multimodal. Existing sensor-only numbers are validation metrics only.
7. Add unit tests using mock 64-D and 768-D arrays so implementation proceeds before Rafaz finishes.
8. Add a command to train/evaluate after real NLP artifacts are merged. Do not run official FD001 test evaluation before final model selection.

## Independence boundaries
- Do not import Rafaz's NLP implementation; mock embeddings are for tests only.
- Do not modify Jacil's IoT files or the frozen LSTM artifacts.
- Never use official test targets for training, validation, checkpoint selection or architecture choice.
- Do not join by array order alone. Validate split and engine metadata.
- Discuss any proposed interface changes before changing shared contracts.

## Suggested files
```text
models/multimodal_fusion.py
training/train_multimodal.py
training/fusion_dataset.py
training/fusion_metrics.py
scripts/verify_fusion.py
scripts/evaluate_multimodal.py
tests/test_fusion_shapes.py
tests/test_fusion_alignment.py
tests/test_missing_text.py
artifacts/models/fd001_multimodal/
artifacts/metrics/fd001_multimodal/
```

## Tests
- Input shapes `[batch,64]` and `[batch,768]` validated.
- Output shape consistently `[batch]` or `[batch,1]`.
- No NaN/Inf outputs or loss.
- Duplicate/missing metadata keys and split mismatches detected.
- No train/validation engine overlap.
- Missing-text windows remain represented.
- Best checkpoint matches lowest validation RMSE.
- Mock pipeline runs without Rafaz's code or a BERT download.
- Official test labels are not accessed during development/model selection.

## Done when
Fusion model and tests run with mock embeddings; pipeline is ready to accept Rafaz's real embeddings. Clearly label mock outputs as smoke-test results. Include commands, architecture summary, test results and limitations.

## Git handoff
```powershell
git add <your-files>
git commit -m "Add multimodal fusion model and evaluation harness"
git push -u origin feature/wahid-multimodal-fusion
```
Open a PR; do not merge directly into `main`.
