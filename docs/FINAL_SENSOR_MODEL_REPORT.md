# Final sensor model report

The frozen FD001 baseline uses 17 ordered features, 30 consecutive cycles, train-fitted StandardScaler and the existing LSTM checkpoint. Runtime validation rejects missing/unexpected/forbidden fields, non-finite values, duplicate or gapped cycles, mixed units and short windows. The API returns `sensor_only`, model ID, unit, window bounds and prediction semantics. Official test metrics remain historical artifact evidence and were not overwritten or used for training/validation in this phase.
