# Wahid dependency setup

## Verified repository declarations

`requirements.txt` declares `fastapi>=0.100,<1`, `uvicorn[standard]>=0.23,<1`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `torch`, and `pytest`; `transformers` is optional and commented out. `frontend/package-lock.json` is authoritative for the JavaScript install. The Tauri manifest is under `src-tauri/Cargo.toml`.

The verification environment used Python 3.11.9, FastAPI 0.128.8, Uvicorn 0.50.0, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, joblib 1.5.3, PyTorch 2.12.1+cpu, pytest 9.1.1, Node/npm from the installed Windows toolchain, and Rust/Cargo. These are observations, not a committed lockfile. CUDA is unavailable (`torch.cuda.is_available() == False`).

## Isolated setup (PowerShell)

```powershell
cd C:\Users\moham\Desktop\IOT_Integrated
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Install a CUDA-specific PyTorch build only after confirming the target GPU, supported Python/CUDA matrix, and an approved wheel index. Do not upgrade the broad requirements blindly. `transformers` is required only for the approved 768-D text embedding implementation; first approve the exact model/tokenizer revision, licence, storage and offline policy before installing or downloading weights.

```powershell
cd frontend
npm ci
npm run build
cd ..
cargo check --manifest-path src-tauri/Cargo.toml
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The current environment's full suite result is recorded in `JACIL_FINAL_PRE_HANDOVER_REPORT.md`. MQTT/InfluxDB extras are not declared in the core requirements; install `paho-mqtt` or `influxdb-client` only for an authorized isolated service test. MongoDB is not required by the current implementation.
