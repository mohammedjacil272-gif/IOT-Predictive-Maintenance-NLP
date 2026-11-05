# Azure PdM Phase 8–9 model card and limitations

## Model

- Model: Phase 7 Extra Trees classifier with engineered telemetry aggregates and machine metadata.
- Lookback: 24 contiguous hourly observations.
- Classes: `none`, `comp1`, `comp2`, `comp3`, `comp4`, `multi_failure`.
- Thresholds: frozen from chronological validation only.
- Artifact: external `phase7_outputs/selected_extra_trees/model_tuned.pkl`.

## Intended use

Offline, reproducible demonstration on the Microsoft Azure PdM simulated benchmark. The output is a model prediction, not confirmation of a real-world failure.

## Limitations

- Source timestamps are timezone-naive benchmark observations; no UTC/IST or ingestion-time meaning is inferred.
- Errors and maintenance are not runtime inputs because independent availability evidence is absent.
- Failure records are target-only and are never used in runtime features.
- `multi_failure` remains rare and threshold-dependent.
- MQTT/InfluxDB live integration is not demonstrated or connected.
- Structured events are not genuine technician narratives; no NLP/BERT branch is claimed.
- The FD001 RUL/fusion system remains isolated and is not reused by this classifier.
