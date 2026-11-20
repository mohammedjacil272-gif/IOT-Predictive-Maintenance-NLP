# Multimodal training readiness

## Final decision

**NO-GO for genuine multimodal training/evaluation.** Software interfaces and synthetic fixture checks are ready for Wahid. Training remains blocked by data authorization, identity/time alignment, split, provenance, and model-asset gates.

| Gate | Evidence | Status |
|---|---|---|
| Sensor baseline | Six frozen FD001 hashes match; 30x17 input and 64-D embeddings exist | READY for read-only comparator use |
| NLP interface | Synthetic fixture, rule-based outputs, and 768-D fixture shape validate | READY for interface tests only |
| DistilBERT runtime | `transformers` and immutable local weights/revision are unavailable | BLOCKED |
| Genuine technician narratives | None authorized or present | BLOCKED |
| Azure PdM files | Filename audit only; simulated benchmark; no authorized acquisition evidence | BLOCKED |
| Identity mapping | No approved `equipment_id`/namespace → `unit_id` crosswalk | BLOCKED |
| Time/cycle alignment | Technician timestamps and FD001 cycles are not established as equivalent | PENDING OWNER DECISION |
| Split compatibility | Synthetic `fixture`; no shared real train/validation/test entity split | BLOCKED |
| Provenance/licensing | No genuine source manifest, permission, or retention evidence | BLOCKED |
| Missing/multiple logs | Provisional implementation exists; production policy not approved | PENDING APPROVAL |
| Leakage controls | Contract and validators exist; no genuine dataset to validate | NOT VALIDATED |
| Training authorization | No approved objective, target/horizon, or run authorization | BLOCKED |
| Evaluation authorization | No approved paired held-out set or comparator protocol | BLOCKED |

## Contract

Sensor records require `unit_id`, `cycle`, `split`, 64-D `sensor_embedding`, and `rul` only for permitted supervised data. Text records require unique `log_id`, approved `unit_id`, `cycle`, `split`, and finite 768-D `text_embedding`. Joins must use metadata, never row position. Provenance must survive every adapter. Missing text must remain explicit (`has_log` or approved equivalent); do not invent text, mappings, timestamps, labels, or metrics.

Before a genuine run, owners must approve the source/licence, crosswalk, time/cycle rule, cutoff/horizon, missing/multiple-log policy, entity-level split, target provenance, leakage checks, model/tokenizer revisions, seed/config, and acceptance metrics. The frozen sensor-only comparator must use its existing artifacts unchanged.
