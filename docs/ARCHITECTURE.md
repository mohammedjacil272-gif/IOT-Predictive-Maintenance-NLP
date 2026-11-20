# Integrated architecture

The project has three distinct model modes and one application boundary:

```text
FD001 sensor data -> validation -> 30-cycle windows -> LSTM -> RUL/64-D embedding
Technician logs -> schema/provenance -> normalization -> rule NER/severity -> optional DistilBERT embedding
Sensor + text representations -> approved alignment/provenance -> fusion model -> fused output
UI -> FastAPI service -> existing Python components -> local artifacts or configured IoT services
```

`data/` contains FD001 files. `preprocessing/`, `training/`, `models/`, `nlp/`, and `iot/` remain independently callable. `app/main.py` is a thin service layer and does not load models in route definitions until inference is requested. The static UI never executes models directly.

Sensor-only FD001 RUL and SCANIA Component X repair-endpoint classification are separate tasks and must not be combined in one metric or visualization. Fusion remains gated by real paired data, mapping, time semantics, approvals, and a trained checkpoint.
