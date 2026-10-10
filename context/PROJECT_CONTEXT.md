# Project Context

## Project

IoT + ML + NLP Based Multimodal Predictive Maintenance

## Goal

The goal of this project is to build a predictive-maintenance system that combines industrial sensor time-series data with unstructured technician maintenance logs.

The system is planned to learn from two complementary sources:

- Structured sensor readings collected over machine operating cycles.
- Unstructured technician maintenance logs that may contain descriptions of faults, symptoms, repairs, warnings, and severity indicators.

The final system should support predictive maintenance by estimating equipment health, identifying failure risk, and comparing a sensor-only baseline against a later multimodal model.

## High-Level Architecture

The planned architecture has two branches that meet in a future multimodal fusion stage.

Sensor branch:

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
    |
Multimodal Fusion
```

Text branch:

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
    |
Multimodal Fusion
```

The sensor branch will first establish a reliable baseline. The text branch and fusion model will be developed only after the preprocessing and sensor-only modeling stages are complete.

## Planned Development Phases

### Phase 1: C-MAPSS FD001 Preprocessing

Build a clean and reproducible preprocessing pipeline for the NASA C-MAPSS FD001 dataset.

The verified project dataset files are stored directly under `Data/`. The current FD001 experiment uses:

```text
Data/train_FD001.txt
Data/test_FD001.txt
Data/RUL_FD001.txt
```

The `Data/` directory also contains FD002, FD003, and FD004 train, test, and RUL files for possible later experiments. Do not rename the `Data/` directory.

### Phase 2: Sensor-Only LSTM Baseline

Train an LSTM model using only sensor time-series windows. This model will become the official baseline for later comparison.

### Phase 3: Sensor Embedding Extraction

Extract learned sensor embeddings from the trained LSTM pipeline for use in future multimodal fusion.

### Phase 4: Technician-Log NLP Pipeline

Create a pipeline for cleaning and processing technician maintenance logs.

### Phase 5: BERT Text Embeddings

Generate text embeddings from technician logs using BERT or a suitable transformer model.

### Phase 6: Multimodal Fusion

Combine sensor embeddings and text embeddings to improve predictive-maintenance performance.

### Phase 7: IoT, MQTT, InfluxDB Integration

Integrate the system with IoT-style data flow, MQTT messaging, InfluxDB storage, and a real-world demonstration setup.

### Phase 8: Evaluation and Comparison

Evaluate the sensor-only baseline and multimodal model using clear, reproducible metrics. Do not fabricate metrics or claim results unless they were actually produced by executed experiments.

## Technology Plan

The planned technology stack includes:

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- HuggingFace Transformers
- spaCy
- MQTT
- InfluxDB

These technologies should not be installed or implemented until explicitly required by the current phase.
