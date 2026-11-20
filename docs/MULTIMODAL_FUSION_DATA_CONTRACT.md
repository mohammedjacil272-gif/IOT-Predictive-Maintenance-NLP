# Multimodal fusion data contract v1

Sensor rows use the frozen FD001 30-cycle windows and 64-D LSTM representation; `unit_id` and `window_end_cycle` are metadata keys, not cross-dataset identity evidence. Text rows carry `log_id`, canonical entity ID, normalized text, event/availability timestamps, provenance, synthetic/real namespace, NER/severity, embedding identity and 768-D vector.

Production alignment requires an approved common entity namespace and mapping, point-in-time availability no later than the prediction cutoff, explicit multiple-log and missing-modality policies, and entity-disjoint splits. The target must be one approved definition; FD001 RUL and an unrelated repair classification cannot be mixed. The current implementation rejects missing provenance, invalid dimensions, future records, duplicates, namespace mismatch and split leakage.
