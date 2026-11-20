# Desktop prerequisite report

Status: **BLOCKED BEFORE NATIVE BUILD**

This report records the toolchain state observed in the current Windows environment. No ML model, dataset, checkpoint, or application database was modified.

| Component | Observed state | Consequence |
|---|---|---|
| Node.js | `v22.19.0` | Present; frontend dependency compatibility still needs verification. |
| npm | `11.12.0` | Present; registry lookup works, but install stalled before producing `node_modules`. |
| npm registry | `https://registry.npmjs.org/` | `npm view react version` succeeded. |
| Rust/Cargo | Missing | Tauri compilation and packaging cannot run. |
| rustup | Missing | Rust toolchain is not available for per-user setup. |
| Tauri CLI | Missing | Tauri build commands cannot run. |
| PyInstaller | Missing | Python sidecar executable cannot be packaged. |
| Visual Studio C++ tools | Not verified as installed | Required by the Windows Rust/Tauri MSVC target. |
| WebView2 | Not verified as installed | Required by the packaged Windows runtime. |

## Required next commands after prerequisite approval

Install Rust from the official Rust distribution using the MSVC toolchain, install Visual Studio Build Tools with Desktop development with C++, install PyInstaller in the project virtual environment, then from `frontend/` run:

```powershell
npm install --no-audit --no-fund
npm run build
```

After that, package and test the backend sidecar before running `cargo tauri build`.

No installation or model download was performed by this task.
