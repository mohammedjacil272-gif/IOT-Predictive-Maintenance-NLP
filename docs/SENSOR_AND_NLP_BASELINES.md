# Sensor and NLP baselines

The frozen FD001 LSTM remains an independent sensor-only baseline with its
existing 30×17 input contract and protected artifacts. EDP sensors will require
their own feature meanings, scaler, windows, and target; FD001 preprocessing
will not be reused across domains.

The current rule-based NLP path supports normalization, entities, severity, and
provenance. Transformer inference remains unavailable without verified local
weights and a suitable approved text corpus. No EDP sensor or text baseline was
trained because the raw workbooks were not obtained and inspected.
