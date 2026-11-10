param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$DataDirectory
)

$ErrorActionPreference = 'Stop'
$exePath = (Resolve-Path -LiteralPath $Executable).Path
$dataPath = [IO.Path]::GetFullPath($DataDirectory)
if (-not (Test-Path -LiteralPath (Join-Path $dataPath '.assistant.sqlite3') -PathType Leaf)) {
    throw "Demo data is not prepared at $dataPath. Run scripts/prepare_demo.py first."
}
$env:IOT_APP_DATA_DIR_OVERRIDE = $dataPath
Start-Process -FilePath $exePath -WorkingDirectory (Split-Path -Parent $exePath)
Write-Output "Started MaintainIQ demo with isolated data: $dataPath"
