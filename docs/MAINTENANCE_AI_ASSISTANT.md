# MaintainIQ maintenance AI assistant

The assistant is local-first and Ollama-compatible. `assistant/gateway.py`
reports model availability and performs bounded requests without downloading
weights. `assistant/retrieval.py` is a deterministic lexical, approved-root
index; it is not a neural knowledge claim. `assistant/health.py` returns
evidence and unknowns, and `assistant/confirmation.py` gates observation writes.

The chat boundary now explicitly separates evidence, predictions,
recommendations, and unknowns. Retrieved documents and technician text are
untrusted data, so prompt-injection instructions inside them are not treated as
commands. The model gateway has no destructive tools. Record changes require a
separate confirmation endpoint and are not claimed as model actions.

Ollama remains optional/unavailable unless `IOT_ASSISTANT_MODEL` names an
already-installed local model and the loopback service is running. No model
quality or grounding performance metric is claimed without an actual local
evaluation corpus.
