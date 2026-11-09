# Windows packaging

Install Rust/Cargo, the Tauri CLI, Visual Studio C++ Build Tools and WebView2 prerequisites. Build the frontend, build the Python sidecar, place the target-suffixed executable in `src-tauri/binaries/`, then run `cargo tauri build`. The configured target is a per-user NSIS installer. This checkout has no Rust/Cargo or PyInstaller, so the installer is not currently built or verified.
