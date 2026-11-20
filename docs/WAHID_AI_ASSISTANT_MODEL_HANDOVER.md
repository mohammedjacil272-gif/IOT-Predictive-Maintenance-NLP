# Wahid AI assistant and model handover

## Starting point

The project is a Tauri 2 / React + Vite / TypeScript desktop shell over a FastAPI backend. The backend already exposes local health, status, model availability, frozen FD001 sensor inference, rule-based technician-log analysis, sessions/messages, confirmations, attachments, machine registry, diagnostics, and a fail-closed fusion readiness API. The frontend build passes and the Python suite passes.

The current assistant is orchestration/storage/NLP foundation, not a conversational LLM. No conversational model has been downloaded or fine-tuned, and no retrieval index is claimed.

## Contracts to preserve

- Sensor inference input is a validated window with 30 cycles and 17 selected FD001 features; live requests never receive RUL labels.
- Frozen sensor representation is 64-D.
- Planned text representation is 768-D DistilBERT with attention-mask-aware mean pooling. The local fixture is synthetic; its exact model revision is not established.
- Fusion joins by metadata `(unit_id, cycle)`, never array position. Preserve `split`, `log_id`, provenance, timestamps, and explicit missing-log state.
- Do not rename `equipment_id` to `unit_id`, convert timestamps to cycles, or relabel `fixture` as train/validation without owner evidence.

## Wahid's sequence

1. Create the isolated environment using `docs/WAHID_DEPENDENCY_SETUP.md` and run the current regression checks.
2. Read `AI_ASSISTANT_FEATURE_MATRIX.md`, `context/MULTIMODAL_FUSION_DATA_CONTRACT.md`, and `MULTIMODAL_TRAINING_READINESS.md`.
3. Review the implemented Ollama gateway, choose the approved local model/configuration, and extend timeout/retry/context/prompt policy as needed.
4. Extend the source-grounded document index with approved embedding/PDF providers only after source and dependency review; preserve chunk metadata, indexing/versioning, retrieval, and citations.
5. Add allowlisted tool schemas, input validation, permission checks, confirmations, bounded timeouts, and secret-safe logging.
6. Integrate sensor/NLP results while preserving provenance and blocked states.
7. Only after Rafaz/Jacil approvals and valid paired data: implement the fusion run, leakage controls, baseline comparator, evaluation, and reproducible report.

## Acceptance criteria

No invented citations, credentials, mappings, labels, or performance claims. Tests must cover gateway unavailable/timeout/malformed response, retrieval missing-source behavior, tool rejection/confirmation, attachment limits, session restoration, sensor validation, NLP provenance, missing/multiple logs, and metadata-based fusion alignment. Training requires an approved dataset manifest, split, objective, seed, model/tokenizer revision, hashes, command, environment, metrics, and held-out evaluation.

Conversational LLM fine-tuning is a separate decision from multimodal RUL training and is not automatically required.
