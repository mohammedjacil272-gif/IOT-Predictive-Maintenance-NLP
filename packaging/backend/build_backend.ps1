param([string]$Python = "python")
$ErrorActionPreference = "Stop"
& $Python -m PyInstaller --clean --noconfirm iot-backend.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
