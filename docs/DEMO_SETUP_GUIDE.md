# MaintainIQ portable Windows demo setup

This guide prepares an isolated, provenance-labelled demonstration. It never writes the repository `.assistant.sqlite3` or `.local_history.json`.

## What the demo proves

The demo uses 30 existing rows from `data/test_FD001.txt`, unit 1, cycles 1–30. They remain FD001 benchmark data and are marked `fd001_benchmark_demo_fixture` plus `is_synthetic: true`. The technician note is explicitly synthetic. This is an evidence-assisted multimodal predictive-maintenance workflow: an independent LSTM result and rule-based NLP findings are supplied to the optional local assistant. It is not a trained multimodal fusion model or live industrial telemetry.

## Requirements

- Windows 10/11 with PowerShell.
- Python 3.11+ and the dependencies in `requirements.txt` for source/API setup.
- For the packaged app: WebView2 and the supplied Tauri executable plus its matching sidecar/installer release asset.
- Optional Qwen explanation: Ollama running locally with model `qwen2.5:3b` installed. The application never downloads it automatically.

## Packaged release assets

The current development-machine package is intentionally kept outside normal Git history because the sidecar and installer are each about 375 MB. Obtain the matching installer from the project GitHub Release when published, or copy the existing local installer and verify its hash before use:

Release: https://github.com/mohammedjacil272-gif/IOT-Predictive-Maintenance-NLP/releases/tag/v0.1.0-demo

- Installer: `ForgeAI Industrial Console_0.1.0_x64-setup.exe`
- Current local SHA-256: `A46410F5D66F4C3B68CA1D397FF2D7CD2B1C623065D28E128691FAC5F74B854F`
- Bundled sidecar SHA-256: `0AAE94BE888DC7D098B1D4E711BE296716B17F5B661BC5663C9AF650F46FB103`

Do not distribute an installer whose hash has not been checked against the release notes. The source repository does not require the 375 MB sidecar binary for ordinary review or test runs; a Tauri rebuild requires placing the target-suffixed sidecar under `src-tauri/binaries/` first.

## Prepare isolated demo data

From the repository root:

```powershell
py -3 -m pip install -r requirements.txt
py -3 scripts/prepare_demo.py --data-dir "$env:LOCALAPPDATA\MaintainIQ-Demo-Data"
```

The command refuses to overwrite an existing application database or history file. Choose a new directory if it reports existing data. It prints the actual LSTM prediction, model provenance, benchmark source, and fusion status.

## Optional Qwen setup

Check whether Ollama is available and the model is installed:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
ollama list
```

If approved, install the model manually with `ollama pull qwen2.5:3b`. If Ollama or the model is unavailable, the demo still shows the LSTM and NLP results and labels the assistant explanation unavailable.

For source/API runs, configure the local process without committing a `.env` file:

```powershell
$env:IOT_ASSISTANT_MODEL_PROVIDER = 'ollama'
$env:IOT_ASSISTANT_MODEL = 'qwen2.5:3b'
$env:IOT_ASSISTANT_MODEL_BASE_URL = 'http://127.0.0.1:11434'
```

## Launch

For the packaged executable and a prepared data directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/launch_demo.ps1 `
  -Executable 'path\to\iot-predictive-maintenance.exe' `
  -DataDirectory "$env:LOCALAPPDATA\MaintainIQ-Demo-Data"
```

The packaged app starts its bundled `iot-backend.exe` on a dynamic loopback port and creates a per-launch bearer token. Do not use port 8000 or an existing service port for this demo.

For a source/API run without Tauri:

```powershell
$env:IOT_INTEGRATED_ROOT = "$env:LOCALAPPDATA\MaintainIQ-Demo-Data"
$env:IOT_ASSISTANT_MODEL_PROVIDER = 'ollama'
$env:IOT_ASSISTANT_MODEL = 'qwen2.5:3b'
py -3 -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

## Demonstration

1. Open **Machines** and select `MAINTAINIQ-DEMO-001`.
2. Open **Assistant Workspace** and select the same machine.
3. Click **ANALYZE MACHINE**.
4. Read the independent LSTM RUL, technician findings, provenance labels, assistant status, and `not_validated` fusion status.

The numerical value is calculated at analysis time from the existing checkpoint and scaler. It is never hardcoded. A missing or invalid sensor sequence produces no machine-specific RUL. NLP-only evidence never produces a numerical RUL.

## Troubleshooting

- No demo machine: rerun `prepare_demo.py` with a new empty data directory and launch with that same directory.
- No RUL: verify the selected app resources include `best_model.pt` and `scaler.joblib`; do not retrain or replace them.
- Assistant unavailable: check `ollama list` and the local Ollama service; the independent results remain valid for this demo.
- Port or authentication errors: close only the demo app you started and relaunch. Do not stop an existing backend on port 8000 or 18765.
