# Coding Rules

## Core Engineering Rules

- Keep the project modular, reproducible, readable, and well documented.
- Use meaningful filenames.
- Keep paths configurable.
- Make important parameters configurable.
- Prefer minimal necessary dependencies.
- Avoid unnecessary abstraction.
- Keep preprocessing independent from model code.
- Do not implement future phases prematurely.
- Do not install packages unless explicitly instructed.
- Do not create dataset files unless explicitly instructed.

## Data Leakage Rules

- Never fit transformations using test data.
- Fit normalization, scaling, imputation, or feature transforms only on training data.
- Apply training-fitted transforms to validation or test data.
- Do not use final test labels to influence preprocessing decisions.
- Do not compute global statistics across train and test combined.
- Preserve temporal ordering when building time-series windows.
- Avoid any operation that allows future cycles to influence earlier-cycle predictions.

## Feature Handling Rules

- Do not silently remove features.
- Keep sensor feature selection explicit and configurable.
- Document any excluded features and the reason for exclusion.
- Validate expected columns before processing.
- Fail clearly when dataset structure does not match expectations.

## C-MAPSS FD001 Rules

- Expected files are:

```text
Data/train_FD001.txt
Data/test_FD001.txt
Data/RUL_FD001.txt
```

- Expected canonical columns are:

```text
unit_id
cycle
setting_1
setting_2
setting_3
sensor_1 ... sensor_21
```

- Training RUL must be computed per engine:

```text
RUL = maximum cycle of that engine - current cycle
```

- Test RUL values from `RUL_FD001.txt` must be associated with the correct test engines.
- Sliding windows must preserve each engine's cycle order.
- Default sliding-window size is 30 cycles unless configured otherwise.

## Validation Rules

- Validate assumptions with actual dataset inspection.
- Check file existence before loading.
- Check column counts and expected schema.
- Check that engine IDs and cycles behave as expected.
- Check that generated arrays have shapes suitable for future PyTorch LSTM use.
- Do not claim that code was tested unless the test or command was actually executed.
- Do not fabricate metrics.

## Phase Discipline

The current phase is only C-MAPSS FD001 preprocessing.

Do not implement:

- LSTM model code.
- Sensor embedding extraction.
- NLP preprocessing.
- NER.
- Severity scoring.
- BERT embeddings.
- Multimodal fusion.
- MQTT integration.
- InfluxDB integration.
- Evaluation dashboards.

Future phases should be added only when explicitly requested.
