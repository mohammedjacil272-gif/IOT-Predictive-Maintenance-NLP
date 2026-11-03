# Current Implementation Phase

## Status

The repository has progressed beyond the original preprocessing-only phase
described in early project notes. Current implementation and actual blockers
are summarized in the root [README](../README.md) and
[architecture](ARCHITECTURE.md).

Implemented code includes FD001 preprocessing and an LSTM baseline, NLP
preprocessing/entity/severity/embedding artifact generation, a provenance-aware
fusion adapter and MLP, and synthetic tests/verifiers. Existing sensor baseline
artifacts are protected. The checked-in baseline checkpoint files are absent.

## Current safe work

- Maintain and test the software contracts with synthetic fixtures.
- Preserve provenance and entity-level `log_id` ownership.
- Keep synthetic namespaces distinct from genuine namespaces.
- Inspect existing artifacts read-only; do not regenerate frozen baseline data,
  embeddings, scaler, metrics, or checkpoints.

## Blocked work

Genuine fusion training and predictive evaluation remain blocked pending
permissioned paired data, verified equipment and cycle mapping, prediction-time
availability evidence, approved target/horizon and split policies, explicit
missing/multiple-log decisions, verified model/checkpoint lineage, and explicit
training authorization. Synthetic tests do not satisfy these requirements.

MQTT, InfluxDB, MongoDB, Azure, Ollama, spaCy and live IoT integrations are not
implemented in this checkout. They require separately approved interfaces,
configuration and isolated tests. No service connection or cloud execution has
been performed.
