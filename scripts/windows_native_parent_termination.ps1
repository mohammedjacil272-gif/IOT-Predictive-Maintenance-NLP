param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$Sidecar
)

$ErrorActionPreference = 'Stop'
$data = Join-Path ([IO.Path]::GetTempPath()) ('IOT Native Lifecycle ' + [guid]::NewGuid().ToString())
$work = Join-Path ([IO.Path]::GetTempPath()) ('IOT Native Work ' + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $data, $work | Out-Null
$exePath = (Resolve-Path -LiteralPath $Executable).Path
$sidecarPath = (Resolve-Path -LiteralPath $Sidecar).Path
$si = [Diagnostics.ProcessStartInfo]::new()
$si.FileName = $exePath
$si.WorkingDirectory = $work
$si.UseShellExecute = $false
$si.Environment['IOT_APP_DATA_DIR_OVERRIDE'] = $data
$parent = [Diagnostics.Process]::Start($si)
try {
    Start-Sleep -Seconds 8
    if ($parent.HasExited) { throw "native executable exited early with code $($parent.ExitCode)" }
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $sidecarPath -and $_.ParentProcessId -eq $parent.Id }
    if (-not $children) { throw 'owned sidecar child was not observed' }
    $childPids = @($children | Select-Object -ExpandProperty ProcessId)
    "parent_pid=$($parent.Id) sidecar_pids=$($childPids -join ',') data=$data"
    # Controlled crash simulation: terminate only the exact parent PID created above.
    $parent.Kill(); $parent.WaitForExit(10000) | Out-Null
    Start-Sleep -Seconds 3
    $remaining = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $sidecarPath -and $_.ProcessId -in $childPids })
    if ($remaining.Count -ne 0) { throw "orphaned exact sidecar process(es): $($remaining.ProcessId -join ',')" }
    'parent_terminated=True sidecar_orphan=False'
} finally {
    if ($parent -and -not $parent.HasExited) { $parent.Kill(); $parent.WaitForExit(10000) | Out-Null }
    Remove-Item -LiteralPath $data -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
