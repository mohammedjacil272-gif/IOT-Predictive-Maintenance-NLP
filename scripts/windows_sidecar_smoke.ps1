param(
    [Parameter(Mandatory = $true)][string]$Executable
)

$ErrorActionPreference = 'Stop'
$data = Join-Path ([IO.Path]::GetTempPath()) ('IOT Desktop Smoke ' + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $data | Out-Null
$port = Get-Random -Minimum 28000 -Maximum 48000
$tokenBytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
try { $rng.GetBytes($tokenBytes) } finally { $rng.Dispose() }
$token = ([BitConverter]::ToString($tokenBytes) -replace '-', '').ToLowerInvariant()
$si = [Diagnostics.ProcessStartInfo]::new()
$si.FileName = (Resolve-Path -LiteralPath $Executable).Path
$si.Arguments = "--host 127.0.0.1 --port $port"
$si.WorkingDirectory = $data
$si.UseShellExecute = $false
$si.RedirectStandardOutput = $true
$si.RedirectStandardError = $true
$si.Environment['IOT_INTEGRATED_ROOT'] = $data
$si.Environment['IOT_API_TOKEN'] = $token
$process = [Diagnostics.Process]::Start($si)
try {
    $base = "http://127.0.0.1:$port"
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod "$base/api/health" -TimeoutSec 2
            if ($health.status -eq 'ok') { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw 'sidecar readiness timeout' }
    try { $unauthenticated = (Invoke-WebRequest "$base/api/status" -UseBasicParsing).StatusCode }
    catch { $unauthenticated = [int]$_.Exception.Response.StatusCode }
    $headers = @{ Authorization = "Bearer $token" }
    $authenticated = (Invoke-WebRequest "$base/api/status" -Headers $headers -UseBasicParsing).StatusCode
    $models = Invoke-RestMethod "$base/api/models" -Headers $headers
    $log = Invoke-RestMethod "$base/api/technician-log/analyze" -Method Post -Headers $headers -ContentType 'application/json' -Body '{"log_id":"smoke","text":"Critical bearing vibration"}'
    "health=$($health.status) unauth=$unauthenticated auth=$authenticated sensor=$($models.sensor_only.status) nlp=$($log.model_status) data=$data pid=$($process.Id)"
} finally {
    if ($process -and -not $process.HasExited) { $process.Kill(); $process.WaitForExit(10000) | Out-Null }
    # PyInstaller one-file mode may leave an exact-path bootstrap child while
    # the wrapper exits. Reap only descendants with the exact executable path.
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $children = Get-CimInstance Win32_Process | Where-Object {
            $_.ExecutablePath -eq $si.FileName -and $_.ProcessId -ne $PID
        }
        foreach ($child in $children) { Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue }
        if (-not (Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $si.FileName -and $_.ProcessId -ne $PID })) { break }
        Start-Sleep -Milliseconds 250
    }
    if (Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $si.FileName -and $_.ProcessId -ne $PID }) { throw 'sidecar process tree did not exit' }
    $exitCode = $process.ExitCode
    for ($attempt = 0; $attempt -lt 10 -and (Test-Path -LiteralPath $data); $attempt++) {
        Remove-Item -LiteralPath $data -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $data) { Start-Sleep -Milliseconds 250 }
    }
    "sidecar_exit=$exitCode temp_removed=$(-not (Test-Path -LiteralPath $data))"
}
