# Windows deployment status

Status: local backend/frontend startup documented; clean-machine installer
verification not completed in the current environment.

Python and Node dependencies are available locally. Rust/Cargo are not on PATH,
so Tauri compilation and NSIS packaging cannot be verified here. The desktop
shell uses a dynamic loopback backend and separates read-only resources from
per-user writable data. Install Rust/Cargo and run the existing Windows smoke
and Tauri commands on a clean Windows machine before claiming installer
readiness.
