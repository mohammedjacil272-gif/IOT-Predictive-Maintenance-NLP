# Desktop architecture

Windows-first desktop architecture: Tauri 2 shell, React/TypeScript/Vite frontend, existing Python FastAPI backend as a loopback sidecar, and existing SQLite/ML/NLP services. The frontend uses real backend contracts for machines, assistant analysis, history and fusion readiness. The current checkout contains the shell configuration and frontend source but cannot produce a verified installer until Rust/Cargo, Tauri CLI, WebView2/build tools and PyInstaller are supplied.
