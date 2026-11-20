# Final Wahid handover checklist

## Start immediately

- [ ] Read `WAHID_AI_ASSISTANT_MODEL_HANDOVER.md`, `AI_ASSISTANT_FEATURE_MATRIX.md`, and `MULTIMODAL_TRAINING_READINESS.md`.
- [ ] Create the isolated environment using `WAHID_DEPENDENCY_SETUP.md`.
- [ ] Run `python -m pytest -q` and the gateway/retrieval targeted tests.
- [ ] Configure an already-installed local Ollama model only if approved; run `/api/assistant/model-status`.
- [ ] Exercise session chat with mocked gateway tests before live Ollama testing.
- [ ] Add approved retrieval embedding/PDF support only with source and dependency review.

## Do not start yet

- [ ] Do not download model weights or datasets without authorization.
- [ ] Do not train fusion or create a fusion checkpoint.
- [ ] Do not map equipment IDs or timestamps by inference.
- [ ] Do not call production MQTT topics or InfluxDB buckets.
- [ ] Do not alter frozen FD001 artifacts.

## Required handoff from others

- Jacil: isolated IoT service endpoint/credential procedure and final desktop/live integration evidence.
- Rafaz: authorized genuine narratives, acquisition/licence evidence, immutable DistilBERT metadata, equipment crosswalk, temporal semantics, and provenance manifest.
- Project owner: approved target/cutoff/horizon, split policy, missing/multiple-log policy, training authorization, and evaluation protocol.
