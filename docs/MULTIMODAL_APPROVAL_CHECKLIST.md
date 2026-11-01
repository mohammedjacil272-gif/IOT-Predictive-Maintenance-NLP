# Multimodal Approval Checklist

**Audit basis:** `1ffc5be56b697a6fe3c97df0c32c1b5ed0f5bf3f` (`origin/main` at audit time).

**Status meanings:** `VERIFIED IMPLEMENTATION` describes observed code behavior, not domain approval. `PROPOSED` is a candidate policy. `PENDING APPROVAL` requires an explicit decision. `BLOCKED` means necessary source evidence/artifacts are unavailable. No item authorizes fusion training or performance evaluation.

| # | Decision / gate | Current evidence | Status | Owner / evidence needed | Approval date |
|---:|---|---|---|---|---|
| 1 | Canonical alignment key `(unit_id, cycle)` | Builder joins by key, not array position. | Implemented; production approval pending | Wahid + data owner: confirm the key on genuine paired data. | — |
| 2 | Cycle meaning | Sensor loader maps `label_cycle` (window end) to `cycle`. | Implemented for sensor windows; cross-source semantics pending | Data owner: source definitions and paired examples. | — |
| 3 | Prediction cutoff | Current text contract does not require a cutoff timestamp. | Pending approval | Product/domain owner: define what time the prediction represents. | — |
| 4 | Technician-log availability | No required authoritative `available_at` or equivalent field in current contract. | Blocked | Data owner: provide authoritative source field/auditable evidence and timezone semantics. | — |
| 5 | RUL target | Baseline train target is uncapped `max_cycle(unit) - cycle`; window target is at `label_cycle`. | Baseline formula verified; fusion target proposed | Project/data owner: approve target, cutoff, horizon; preserve baseline. | — |
| 6 | Operational horizon | No finalized deployment horizon established. | Pending approval | Product/domain owner: approve use case and horizon. | — |
| 7 | Genuine paired dataset / permission | Current five-log fixture is synthetic. No genuine pair is established by available evidence. | Blocked | Dataset owner: source, permission/license, approved use. | — |
| 8 | Equipment namespace/mapping | Fixture IDs do not prove real ID equivalence. | Blocked / pending | Data owner: stable IDs, versioned crosswalk and validation. | — |
| 9 | Production provenance schema | Basic interface requires `log_id`, `unit_id`, `cycle`, `split`, `text_embedding`; source/time lineage not required. | Basic interface verified; production schema pending | Wahid + data owner: review schema proposal. | — |
| 10 | Equipment split policy | Baseline uses 80/20 engine train/validation split, seed 42, train-only scaler; loader checks train/validation IDs and excludes official test from training input. | Existing guards verified; paired-data test policy pending | Team: approve the paired-data split and final held-out set. | — |
| 11 | Missing-text policy | Zero 768-D vector and `has_log=False` are implemented; model does not consume `has_log` separately. | Implemented default; production policy proposed | Team + Wahid: approve or revise and decide `has_log` role. | — |
| 12 | Multiple-log policy | Deterministic `log_id` ordering, mean embedding, per-log provenance retention are implemented. | Implemented default; production policy proposed | Team + Wahid: approve temporal eligibility, aggregation and traceability. | — |
| 13 | Integrity/leakage controls | Field/key/split/dimension/finiteness checks exist; point-in-time evidence is missing. | Partially implemented; full policy pending | Wahid + data owner: approve real-source checks and negative tests. | — |
| 14 | Evaluation protocol | Proposed same-target, same-held-out-equipment comparison against frozen baseline; no valid fusion performance result. | Proposed / pending approval | Project lead: approve metrics, selection rule, seeds and reporting. | — |
| 15 | Final held-out test set | Training loader accepts train/validation; no approved paired-data final-test protocol exists. | Pending approval | Team: designate immutable held-out equipment population. | — |
| 16 | Dependency manifest/runtime support | No standard dependency manifest/lockfile found in audited tracked inventory. | Blocked | Maintainer: approve Python support, dependencies, resolver and lock strategy. | — |
| 17 | Text model revision/checksum | Loader names `distilbert-base-uncased` but does not pass immutable `revision`; report lacks asset checksums. | Blocked | NLP owner: approve model/tokenizer revision and hashes. | — |
| 18 | Baseline checkpoint files | Neither `best_model.pt` nor `last_model.pt` found in tracked/on-disk audit inventory. | Blocked | Baseline owner: locate authorized matching checkpoints and verify hashes; do not retrain. | — |
| 19 | IoT acceptance | MQTT/Influx/replay/inference code and software tests exist; live service/device/HIL evidence not established. | Software tests exist; live E2E pending | IoT owner: isolated services, configuration and run evidence. | — |
| 20 | CI evidence | No CI run tied to this SHA is captured in this handoff. | Pending evidence | Maintainer: provide check-run URL/SHA or state no checks exist. | — |

## Gate to fusion training/evaluation

Keep both **BLOCKED** until genuine paired data and its permissions, equipment mapping, shared cycle semantics, point-in-time availability, RUL target/cutoff/horizon, provenance schema, split/held-out policy and missing/multiple-log policy are explicitly approved. Never mark a proposal approved because the code implements it or fixture tests pass.

Related records: [decision register](MULTIMODAL_DECISION_REGISTER.md), [data contract](MULTIMODAL_FUSION_DATA_CONTRACT.md), [provenance schema](MULTIMODAL_PROVENANCE_SCHEMA.md), [evaluation protocol](MULTIMODAL_EVALUATION_PROTOCOL.md), [dependencies](REPRODUCIBILITY_AND_DEPENDENCIES.md), [IoT readiness](IOT_END_TO_END_TEST_READINESS.md), [handoff](MULTIMODAL_FUSION_HANDOFF.md).
