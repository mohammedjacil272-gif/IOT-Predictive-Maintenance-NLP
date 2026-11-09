# Maintenance AI assistant status

The local Ollama-compatible gateway, bounded context, lexical approved-root
retrieval, allowlisted read-only machine-health tool, and confirmation-gated
observation workflow are implemented. Grounding rules treat technician text and
retrieved documents as untrusted evidence and disallow model-driven destructive
actions.

The assistant is not yet demonstrated against genuine EDP data or a live local
Ollama model in this run. It must report missing equipment, unavailable models,
missing logs, tool failures, and unknown confidence rather than inventing facts.
Transformer or LLM quality metrics are not claimed.
