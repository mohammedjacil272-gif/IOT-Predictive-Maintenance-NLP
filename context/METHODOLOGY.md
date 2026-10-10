# Methodology

## Phase 1: Data Preprocessing

Begin with the NASA C-MAPSS FD001 dataset.

The verified FD001 files for the current experiment are:

```text
Data/train_FD001.txt
Data/test_FD001.txt
Data/RUL_FD001.txt
```

The preprocessing methodology should:

- Load `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.
- Validate file existence and dataset structure.
- Assign canonical columns explicitly.
- Inspect assumptions using the actual dataset.
- Compute training RUL per engine using maximum cycle minus current cycle.
- Associate each test engine with its corresponding final RUL from `RUL_FD001.txt`.
- Preserve temporal ordering within each engine.
- Select sensor features explicitly through configuration.
- Track raw available sensor and setting columns, informative or non-constant features, and the final feature set actually used by the model.
- Fit normalization only on training data.
- Apply the training-fitted scaler to test data.
- Generate sliding windows with configurable parameters.
- Use a default window size of 30 cycles.
- Produce arrays suitable for future PyTorch LSTM input.

Verified FD001 inspection facts:

- `Data/train_FD001.txt` has 20,631 rows, 26 columns, 100 engines, sequential engine IDs 1-100, sequential cycles within each engine, and cycle range 1-362.
- `Data/test_FD001.txt` has 13,096 rows, 26 columns, 100 engines, sequential engine IDs 1-100, sequential cycles within each engine, and cycle range 1-303.
- `Data/RUL_FD001.txt` has 100 values, RUL range 7-145, and corresponds to the 100 FD001 test engines by dataset ordering convention.
- FD001 files are whitespace-delimited, headerless, and have no blank rows, malformed-width rows, or missing values found during inspection.
- Trailing whitespace exists and should be handled through whitespace parsing.
- `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19` are constant in both FD001 train and test.
- Constant columns must not be removed unless the preprocessing design explicitly makes and documents that decision.

## Phase 2: Sensor-Only LSTM Baseline

After preprocessing is complete, train an LSTM model using only sensor time-series windows.

The baseline should:

- Use outputs from the preprocessing pipeline.
- Avoid embedding any preprocessing logic directly inside model code.
- Provide a reproducible comparison point for future multimodal experiments.
- Report only metrics that were actually computed.

## Phase 3: Sensor Embeddings

After the baseline model is trained, extract sensor embeddings from the LSTM branch.

These embeddings should represent learned temporal sensor patterns and will later be fused with text embeddings.

## Phase 4: Technician-Log NLP

Develop an NLP pipeline for technician maintenance logs.

The pipeline may include:

- Text cleaning.
- Tokenization.
- Domain-specific entity extraction.
- Named entity recognition.
- Severity scoring.

This phase should remain independent from the sensor preprocessing code.

## Phase 5: Text Embeddings

Use BERT or a suitable transformer model to generate text embeddings from technician logs.

The text embedding process should be reproducible, configurable, and separated from sensor-only modeling.

## Phase 6: Multimodal Fusion

Fuse sensor embeddings and text embeddings to create a multimodal predictive-maintenance model.

The fusion model should be compared against the sensor-only LSTM baseline.

## Phase 7: IoT Demonstration

Integrate IoT-oriented components after the modeling pipeline is validated.

Possible components include:

- MQTT for message transport.
- InfluxDB for time-series storage.
- A real-world or simulated demonstration workflow.

## Phase 8: Evaluation

Evaluate the complete system with clear methodology and reproducible metrics.

Evaluation must:

- Compare sensor-only and multimodal approaches.
- Avoid fabricated results.
- Clearly document datasets, splits, preprocessing, configuration, and executed commands.
- Distinguish planned work from completed experiments.
