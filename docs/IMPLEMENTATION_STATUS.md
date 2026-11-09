# Implementation status

## COMPLETE AND VERIFIED

- FD001 preprocessing, 17-feature/30-cycle contract, engine-disjoint split, training-only scaling, and artifact verification.
- FD001 LSTM baseline loading/inference and sensor embedding extraction.
- Rule-based technician-log preprocessing, NER, severity, and synthetic provenance checks.
- Fusion model shape validation, alignment, provenance, missing-text handling, and synthetic forward-only smoke test.
- Local FastAPI health/status/model/log-analysis endpoints and static dashboard.

## IMPLEMENTED BUT NOT FULLY VERIFIED

- MQTT/Influx adapters: software tests pass; live broker/database not checked.
- Optional DistilBERT embedding adapter: code exists, but `transformers` and local model assets are unavailable here.
- Fusion dataset/training interfaces: contract/synthetic tests pass; genuine training is not authorized or evidenced.

## BLOCKED BY EXTERNAL DEPENDENCY

- Genuine NLP training/evaluation: approved logs, labels, machine mapping, and embedding assets.
- Genuine fusion training/evaluation: paired data, temporal semantics, target/cutoff policy, approval, and checkpoint.

## NOT IMPLEMENTED

- Authentication, production deployment, live-service orchestration, and a production model registry.
