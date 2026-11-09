# Sensor workflow completion

The sensor adapter validates the exact 17 selected FD001 features, one positive unit, finite numeric values, unique consecutive cycles, and at least 30 records. It uses the frozen scaler and checkpoint, takes the latest 30 cycles, and returns unit ID, window bounds, prediction and `sensor_only` mode. Missing fields, labels, malformed values, gaps, duplicate cycles and mixed units fail with explicit errors. Frozen artifacts were not modified.
