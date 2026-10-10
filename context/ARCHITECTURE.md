# Architecture

## Overview

The planned system is a multimodal predictive-maintenance architecture. It will combine structured sensor time-series data with unstructured technician maintenance logs.

Development should proceed incrementally. The sensor branch must be built and validated first, then used as the official baseline before adding NLP and multimodal fusion components.

## Sensor Branch

The sensor branch processes machine operating data from the NASA C-MAPSS FD001 dataset.

Planned flow:

```text
Sensor Data
    |
Preprocessing
    |
Sliding Windows
    |
LSTM
    |
Sensor Embedding
```

Responsibilities:

- Load and validate C-MAPSS FD001 files from `Data/`.
- Compute remaining useful life labels correctly.
- Preserve per-engine temporal order.
- Normalize data without test leakage.
- Generate sliding windows for sequence modeling.
- Train a future LSTM baseline.
- Extract learned sensor embeddings after the baseline is established.

The verified FD001 files are whitespace-delimited, headerless, 26-column files. Positional columns map to engine/unit ID, cycle, three operational settings, and 21 sensor measurements. Some setting and sensor columns are constant in both FD001 train and test, but feature selection must remain explicit and must not silently remove them.

The sensor-only LSTM model will become the official baseline for comparing later multimodal improvements.

## NLP Branch

The NLP branch will process unstructured technician maintenance logs.

Planned flow:

```text
Technician Logs
    |
NLP Preprocessing
    |
NER + Severity Scoring
    |
BERT
    |
Text Embedding
```

Responsibilities:

- Clean and normalize technician maintenance text.
- Identify relevant maintenance entities using NER.
- Estimate severity signals from log content.
- Generate text embeddings using BERT or a suitable transformer model.

This branch should not be implemented until the sensor preprocessing and baseline work are complete.

## Multimodal Fusion

The future fusion stage will combine:

- Sensor embeddings from the LSTM branch.
- Text embeddings from the NLP branch.

Planned flow:

```text
Sensor Embedding ----|
                    |--> Multimodal Fusion --> Predictive Maintenance Output
Text Embedding ------|
```

The fusion model should be evaluated against the sensor-only baseline to determine whether technician logs improve predictive-maintenance performance.

## IoT Integration

The final system may include IoT-style integration using MQTT and InfluxDB.

Planned responsibilities:

- Simulate or ingest streaming sensor data.
- Publish or subscribe to maintenance-relevant data using MQTT.
- Store time-series readings in InfluxDB.
- Demonstrate how the predictive-maintenance pipeline could operate in a real-world environment.

This integration belongs to a later phase and should not affect the preprocessing design except through configurable paths and clean module boundaries.
