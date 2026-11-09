# Local performance benchmark

Execution date: 2026-11-09. Hardware: CPU-only PyTorch 2.15.0+cpu; CUDA was
unavailable. One warmed frozen sensor inference using a 30 x 17 input window,
already-loaded model, and already-loaded scaler measured approximately **3.569
ms** on this development machine. This is a local smoke benchmark, not a
production capacity claim.

Transformer and fusion latency were not measured because approved local
transformer weights and a trained fusion checkpoint are unavailable. MQTT,
InfluxDB, MongoDB, concurrent-request, and clean-machine benchmarks were not
run because live infrastructure and Rust/Tauri tooling are unavailable.

Frontend dependency audit: `npm audit --omit=dev --audit-level=high` reported
`found 0 vulnerabilities`.
