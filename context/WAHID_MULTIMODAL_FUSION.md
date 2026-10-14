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
- Text embeddings: float array/tensor `[batch, 768]`, traceable to `log_id`, `unit_id`, and cycle/window.
- Target: scalar RUL from training/validation split only.
- Output: one unconstrained scalar RUL prediction per joined sample.
- Join by stable metadata keys and alignment rules, never by array index alone.
- Represent missing text explicitly (`has_log` or a documented missing-text vector); do not silently discard those windows.

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
