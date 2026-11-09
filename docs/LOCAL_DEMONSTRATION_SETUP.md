# Local demonstration setup

```powershell
Set-Location C:\Users\moham\Desktop\IOT_Integrated
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Start-Process http://127.0.0.1:8000
```

Use sensor-only and NLP-only flows from the dashboard, then inspect fusion readiness. Optional transformer inference is not required for unrelated flows and is never downloaded automatically. MQTT/InfluxDB are not required for this local demonstration.
