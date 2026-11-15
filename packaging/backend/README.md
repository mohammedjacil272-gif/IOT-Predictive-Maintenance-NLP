# Backend sidecar packaging

# Windows backend sidecar

Run `powershell -ExecutionPolicy Bypass -File packaging/backend/build_backend.ps1 -Python py` from the repository root. The checked-in spec includes only artifacts that exist; if an optional artifact is absent, the API reports it unavailable rather than fabricating a prediction. The current sidecar includes the regenerated FD001 checkpoint and the optional Transformers runtime when installed. Transformer weights are intentionally external and are never bundled or downloaded by the sidecar.

The Tauri configuration expects the target-triple-renamed executable at `src-tauri/binaries/iot-backend-x86_64-pc-windows-msvc.exe`. Copying is a release-packaging step and is intentionally not performed by the build script. The sidecar binds loopback only, receives its writable application-data root through `IOT_INTEGRATED_ROOT`, and is protected by the Tauri-generated bearer token.

For local transformer embeddings, install `requirements-optional-nlp.txt` and point `IOT_NLP_CACHE_DIR` at an approved Hugging Face cache, for example `%USERPROFILE%\.cache\huggingface`. The application uses `local_files_only=True`; if the cache is absent, rule-based NLP remains available and the API reports embeddings as unavailable.
