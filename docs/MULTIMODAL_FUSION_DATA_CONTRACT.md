# Multimodal Fusion Data Contract

**Scope / audit basis:** implementation at `1ffc5be56b697a6fe3c97df0c32c1b5ed0f5bf3f`. This document separates observed behavior from candidate production requirements. It does not approve training or performance evaluation.

## 1. Tensor interface — VERIFIED IMPLEMENTATION

- Sensor record: finite float32 vector `[64]`.
- Text record: finite float32 vector `[768]`.
- Joined batches: sensor `[N,64]`, text `[N,768]`, RUL target `[N]`.
- The model concatenates sensor/text representations and returns one unconstrained scalar prediction per sample.
- Invalid dimensions should fail validation rather than be padded, truncated, or silently reshaped.

## 2. Current required fields — VERIFIED IMPLEMENTATION

### Sensor record

| Field | Current rule |
|---|---|
| `unit_id` | Required integer; boolean values rejected. |
| `cycle` | Required positive integer. In the loader, this is taken from sensor metadata `label_cycle`, the window-end cycle. |
| `split` | Required non-empty string at builder. Multimodal loader restricts source sensor splits to `train`/`validation` and checks unit membership against saved split configuration. |
| `sensor_embedding` | Finite vector of exactly 64 values. |
| `rul` | Finite scalar target copied from baseline `label_rul`; never a live inference feature. |

### Text record

| Field | Current rule |
|---|---|
| `log_id` | Required non-empty unique string. |
| `unit_id` | Required integer; intended same equipment namespace as sensor key. |
| `cycle` | Required positive integer; intended to match sensor `label_cycle`/window end. Genuine cross-source semantics remain unvalidated. |
| `split` | Required non-empty string equal to matched sensor split. |
| `text_embedding` | Finite vector of exactly 768 values. |
| `entities`, severity fields | Optional; supplied entity provenance is checked against owning record; supplied severity score must be finite. |

The builder rejects duplicate sensor keys, duplicate/empty log IDs, unmatched text keys, split mismatch, malformed dimensions, non-finite embeddings and inconsistent entity provenance. It enforces per-unit split consistency in the supplied sensor records. The generic builder does not independently define the full authoritative train/validation/test configuration; the calling training pipeline supplies additional controls.

## 3. Alignment — IMPLEMENTED; REAL-SOURCE SEMANTICS PENDING

The key is `(unit_id, cycle)`, not array position. The sensor-side `cycle` maps from metadata `label_cycle`, meaning the window end. Matching fields with the same names does not prove the underlying systems share semantics. A genuine source owner must validate equipment namespace/crosswalk, cycle meaning, timestamp basis, timezone, precision, and key uniqueness using paired examples.

## 4. Prediction cutoff and availability — BLOCKER

The current text record does not require source creation, event, ingestion, or availability timestamps. The current fixture cannot establish that a technician log was known at prediction time.

Production pairing must exclude text created/available after the approved cutoff. Do not infer `available_at` from event time, work-order open time, or ingestion time without an approved equivalence rule. If an authoritative availability field or auditable equivalent cannot be supplied, keep performance evaluation blocked or exclude those records only under an approved, documented policy.

## 5. RUL target — BASELINE FACT; FUSION APPROVAL PENDING

The baseline training label uses uncapped `RUL(unit, cycle) = max_cycle(unit) - cycle`; each sensor-window target corresponds to its final/label cycle. Official FD001 test-row labels use a separate supplied-final-RUL calculation. The fusion loader maps baseline `label_rul` to `rul`; do not recalculate, cap or alter baseline labels.

Formal approval of the fusion target, prediction cutoff and operational horizon remains pending. Official test labels must not enter training or model selection or be live inference features.

## 6. Missing/multiple log behavior — IMPLEMENTED DEFAULTS, NOT APPROVED PRODUCTION POLICY

- No matched text: retain sensor row, fill zero 768-D text vector, return `has_log=False`.
- Multiple logs at one key: sort deterministically by `log_id`, average embeddings, retain per-log IDs/entities/severity separately.
- The training dataset passes sensor embeddings, text embeddings and RUL target; `has_log` is not currently passed as a separate model input.

The team must approve zero-vector handling, whether `has_log` becomes a model input or remains reporting metadata, the time-window inclusion rule and mean aggregation. Approval requires evidence from genuine data, especially missingness and multiple-log examples.

## 7. Splits and leakage — PARTIALLY IMPLEMENTED

- Baseline preparation uses 80 train engines / 20 validation engines, seed 42, scaler fitted only on training-engine rows, 30-cycle windows, stride 1.
- Multimodal loader accepts only `train` and `validation` sensor metadata, validates each sensor record against the saved split config, and checks train/validation engine sets are disjoint.
- The general fusion builder checks sensor per-unit split consistency and matched text/sensor split equality but does not alone establish the entire approved paired train/validation/test policy.
- Source-aware checks for log availability, post-cutoff text, source lineage and target/future leakage remain pending until genuine source metadata exists.

## 8. Verification scripts and approval state

Relevant files at the audit SHA: `training/fusion_dataset.py`, `training/train_multimodal.py`, `tests/test_fusion_alignment.py`, `tests/test_missing_text.py`, and `scripts/verify_nlp_fd001_compatibility.py`.

See [approval checklist](MULTIMODAL_APPROVAL_CHECKLIST.md), [decision register](MULTIMODAL_DECISION_REGISTER.md), [provenance schema](MULTIMODAL_PROVENANCE_SCHEMA.md), and [evaluation protocol](MULTIMODAL_EVALUATION_PROTOCOL.md).

Do not change the shared contract until Wahid and the data owner agree on the genuine source schema and the project records explicit approval.
