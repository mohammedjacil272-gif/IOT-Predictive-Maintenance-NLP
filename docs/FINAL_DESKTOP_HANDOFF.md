# Final desktop handoff

Frontend source is under `frontend/`; Tauri configuration is under `src-tauri/`; backend packaging guidance is under `packaging/backend/` and `packaging/windows/`. Development backend launch remains `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`. The native Windows application and NSIS installer are **NOT BUILT** in this environment. Install the approved Rust/Tauri/PyInstaller toolchain, build the sidecar, then complete installed-app lifecycle testing before UI release.
