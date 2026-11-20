# Windows packaging

Install Rust/Cargo, the Tauri CLI, Visual Studio C++ Build Tools and WebView2 prerequisites. Build the frontend, build the Python sidecar, place the target-suffixed executable in `src-tauri/binaries/`, then run `cargo tauri build`. The configured target is a per-user NSIS installer. On 2026-11-10 this workflow produced and hashed the x64 NSIS installer; clean-machine installation remains unverified.
