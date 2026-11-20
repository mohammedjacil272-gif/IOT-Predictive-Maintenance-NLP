# Backend sidecar design

The Tauri shell is configured to bundle an `iot-backend` external binary and a per-user NSIS installer. The sidecar must bind loopback only, expose `/api/health`, select a free local port, communicate the port/token to the frontend, and shut down through an owned process handle. The current Rust runtime is a minimal shell and does not yet implement lifecycle supervision; this remains a build-phase blocker rather than a claimed feature.
