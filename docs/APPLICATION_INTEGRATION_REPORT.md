# Application integration report

FastAPI, React, and the local assistant workspace are integrated for the
verified sensor-only/NLP demonstration. Readiness endpoints distinguish missing
models and unrun official evaluation from completed validation. Attachment
types are validated consistently for TXT, CSV, JSON, PDF, JPEG, and PNG.

The frontend production build passes. Rust/Cargo/Tauri checks were not run
because Rust and Cargo are unavailable on PATH in this environment; no Windows
installer or clean-machine desktop launch is claimed.

The local Ollama assistant is optional and never downloads a model. Its chat
prompt treats retrieved documents as untrusted evidence, exposes read-only
grounding metadata, and keeps observation writes behind explicit confirmation.
