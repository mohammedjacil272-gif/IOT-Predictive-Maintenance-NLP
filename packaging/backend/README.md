# Backend sidecar packaging

The Windows sidecar was built experimentally with PyInstaller as `dist/iot-backend.exe` (outside the Tauri bundle). It has not yet been proven in a clean environment, and it is not yet copied into `src-tauri/binaries/` because the Rust/Tauri shell and owned-process lifecycle are still blocked. Before release, the sidecar must bind FastAPI to loopback, use Tauri application-data paths, expose `/api/health`, and exit on application shutdown.
