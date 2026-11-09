param([string]$Python = "python")
$ErrorActionPreference = "Stop"
& $Python -m PyInstaller --name iot-backend --onefile --clean `
  --add-data "app;app" `
  --add-data "assistant;assistant" `
  --add-data "nlp;nlp" `
  --add-data "artifacts/models/fd001_lstm_baseline/best_model.pt;artifacts/models/fd001_lstm_baseline" `
  --add-data "artifacts/data/fd001_lstm_baseline/scaler.joblib;artifacts/data/fd001_lstm_baseline" `
  --add-data "artifacts/embeddings/fd001_lstm_baseline/sensor_embeddings.npy;artifacts/embeddings/fd001_lstm_baseline" `
  --hidden-import "app.main" `
  --collect-submodules "app" `
  --collect-submodules "assistant" `
  --collect-submodules "nlp" `
  packaging/backend/sidecar_entry.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
