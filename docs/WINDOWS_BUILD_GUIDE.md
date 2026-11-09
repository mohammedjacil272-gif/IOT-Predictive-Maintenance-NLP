# Windows build guide

1. Install Rust/Cargo and Tauri CLI.
2. Install approved frontend dependencies in `frontend/`.
3. Install PyInstaller in the approved build environment and run `packaging/backend/build_backend.ps1`.
4. Place the target-suffixed sidecar under `src-tauri/binaries/`.
5. Run `cargo tauri build` from the repository root. The Tauri config runs the frontend build from `../frontend` relative to `src-tauri`.
6. Verify output hashes and run both Windows smoke scripts.

The final local build produced a native executable, target-suffixed sidecar, and NSIS installer. Installed-app workflow and normal user-driven close remain separate verification steps.
