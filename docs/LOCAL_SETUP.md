# Windows local setup

From `C:\Users\moham\Desktop\IOT-MaintainIQ-Audit`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. In a second terminal:

```powershell
npm --prefix frontend ci
npm --prefix frontend run dev
```

No database or broker is required for local registry and technician-log analysis. Sensor prediction remains unavailable until the verified FD001 checkpoint is supplied. To use the already-installed Ollama model, set it before starting the backend: `$env:IOT_ASSISTANT_MODEL = "qwen2.5:3b"`. The backend reads environment configuration at startup. The local MQTT simulation is `python -m iot.mqtt_publisher --dry-run --limit 1`.
