# Windows installation guide

The current build outputs are:

- Native executable: `src-tauri/target/release/iot-predictive-maintenance.exe`
- NSIS installer: `src-tauri/target/release/bundle/nsis/ForgeAI Industrial Console_0.1.0_x64-setup.exe`

The application uses a per-user AppData directory for SQLite, JSON history, and attachments. It binds the bundled sidecar to loopback on a dynamic port with a per-launch bearer token. Fusion is intentionally reported unavailable until approved paired data and a verified checkpoint exist.

The installer is a per-user NSIS package and requires Windows 10/11, WebView2, Rust/Cargo, the MSVC build tools, Node/npm, and the packaged Python sidecar. The sidecar binds only to loopback and uses a per-launch token.

Build from the repository root:

1. Run `npm --prefix frontend install`.
2. Run `powershell -ExecutionPolicy Bypass -File packaging/backend/build_backend.ps1 -Python python`.
3. Place the resulting sidecar at `src-tauri/binaries/iot-backend-x86_64-pc-windows-msvc.exe`.
4. Run `cargo tauri build`.

Use `scripts/windows_sidecar_smoke.ps1 -Executable dist/iot-backend.exe` for the packaged API/auth smoke test and `scripts/windows_native_parent_termination.ps1 -Executable src-tauri/target/release/iot-predictive-maintenance.exe -Sidecar src-tauri/binaries/iot-backend-x86_64-pc-windows-msvc.exe` for the isolated unexpected-parent-termination test. `python scripts/installer_smoke_test.py` installs into a temporary directory, launches the installed executable, and invokes the uninstaller; its immediate post-uninstall directory check is timing-sensitive because NSIS cleanup is asynchronous.

For a quick manual launch, double-click the native executable above. WebView2 is required on Windows 10/11. No model weights are downloaded by the application; optional local transformer resources must be supplied separately if transformer embeddings are needed.
