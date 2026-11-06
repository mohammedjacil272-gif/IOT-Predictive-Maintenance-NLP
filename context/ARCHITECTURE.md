# Architecture and Implementation Status

**Review base:** `origin/main` at `d5b63ee`. This document separates components present in the repository from work that remains proposed, blocked, or unverified. Presence of code and software tests does not establish production readiness or genuine-data performance.

## Implemented components

### Sensor pipeline

The repository contains FD001 preprocessing and dataset/window preparation code, an LSTM baseline, and sensor embedding extraction. Sensor windows use a 30-cycle configuration and 17 selected features in the documented baseline contract; the fusion-facing sensor embedding is 64-dimensional. Existing baseline artifacts must remain immutable. The expected `best_model.pt` and `last_model.pt` checkpoints are absent from the checked-in inventory, so baseline inference requiring those checkpoints cannot be reproduced from this checkout.

### Technician-log NLP

The repository contains log preprocessing, rule-based entity extraction, heuristic severity scoring, transformer embedding generation, and artifact verification. The checked-in five-record technician-log fixture is synthetic. Its entities and embeddings test software interfaces; they are not evidence of genuine technician data or validated labels. The text embedding interface is 768-dimensional. Existing artifact/model lineage does not establish an immutable model and tokenizer revision with complete asset checksums.

### Multimodal fusion

Fusion code aligns sensor windows and text records using metadata, provisionally `(unit_id, cycle)`. It does not use CSV row order or embedding position as a cross-modal key. Entity records must carry their own non-empty `log_id` matching the parent text record in synthetic and genuine namespaces. Genuine inputs have additional source, artifact, timestamp, equipment-mapping, and approval gates.

The implementation exposes explicit missing-log and multiple-log policies and an explicit `has_log` model-feature choice. These are software choices, not approved production policies. Synthetic tests exercise alignment, shapes, provenance, and missing-text behavior. No genuine paired-data fusion training or predictive-performance evaluation is established.

### IoT ingestion and replay

MQTT, InfluxDB, and replay-related code and software/mock tests are present. These tests do not establish live broker/database connectivity, device behavior, or hardware-in-the-loop validation. Such validation requires an approved isolated setup and recorded evidence.

## Planned, pending, or blocked work

- Obtain permissioned genuine paired sensor and technician-log data with auditable lineage.
- Verify authoritative equipment mapping and shared cycle/window semantics before using `(unit_id, cycle)` as a real cross-source key.
- Approve prediction cutoff, timestamp availability semantics, RUL target/horizon, equipment-level splits, and missing/multiple-log policies.
- Establish immutable text-model/tokenizer and artifact provenance, and locate and verify the authentic sensor baseline checkpoints.
- Record separate evidence-backed authorization for genuine fusion training and final held-out evaluation.
- Perform an approved final comparison against the frozen sensor-only baseline on the same eligible held-out population.
- Validate live IoT integrations only with an approved isolated test environment.

Official FD001 test labels must not be used for training or model selection. No official FD001 test evaluation or genuine fusion performance result is established by the reviewed documentation.

## Data flow and trust boundaries

```text
FD001 sensor rows -> preprocessing/windows -> LSTM -> 64-D sensor embeddings
Technician logs -> preprocessing/entities/severity -> 768-D text embeddings
Sensor + log metadata -> namespace/provenance/policy checks -> metadata alignment
```

A synthetic fixture pass proves only the tested software behavior. It does not authenticate source data, prove equipment mapping, establish point-in-time availability, or authorize training. Structural validation does not itself authenticate approvers or external evidence. See the [fusion data contract](MULTIMODAL_FUSION_DATA_CONTRACT.md), [approval checklist](MULTIMODAL_APPROVAL_CHECKLIST.md), [decision register](../docs/MULTIMODAL_DECISION_REGISTER.md), and [evaluation protocol](../docs/MULTIMODAL_EVALUATION_PROTOCOL.md) for detailed contracts and pending decisions.
