# Windows local setup

From `C:\Users\moham\Desktop\IOT_Integrated`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. No database or broker is required for the UI, health, status, or technician-log analysis. Configure MQTT/Influx only when those services are intentionally available. Copy `.env.example` to a local environment file outside Git and never put tokens in source, logs, or UI responses.
