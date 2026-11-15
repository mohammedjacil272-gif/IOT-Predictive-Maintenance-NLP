param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$Sidecar
)

$ErrorActionPreference = 'Stop'
$data = Join-Path ([IO.Path]::GetTempPath()) ('IOT Native Lifecycle ' + [guid]::NewGuid().ToString())
$work = Join-Path ([IO.Path]::GetTempPath()) ('IOT Native Work ' + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $data, $work | Out-Null
$exePath = (Resolve-Path -LiteralPath $Executable).Path
# Tauri copies externalBin to the runtime directory as iot-backend.exe. Keep
# the source-sidecar parameter for CLI compatibility, but never use its
# target-suffixed path for runtime process detection.
$runtimeDirectory = Split-Path -Parent $exePath
$sidecarPath = Join-Path $runtimeDirectory 'iot-backend.exe'
if (-not (Test-Path -LiteralPath $sidecarPath -PathType Leaf)) {
    throw "runtime sidecar not found: $sidecarPath"
}
$sidecarPath = (Resolve-Path -LiteralPath $sidecarPath).Path
function Normalize-ProcessPath {
    param([object]$Value)
    if ([string]::IsNullOrWhiteSpace([string]$Value)) { return $null }
    try {
        return [IO.Path]::GetFullPath([string]$Value).TrimEnd('\', '/').ToLowerInvariant()
    } catch {
        return $null
    }
}
function Get-RuntimeSidecars {
    param([string]$ExpectedPath)
    $matches = @()
    foreach ($process in @(Get-CimInstance Win32_Process)) {
        if ((Normalize-ProcessPath $process.ExecutablePath) -eq $ExpectedPath) {
            $matches += $process
        }
    }
    return $matches
}
$normalizedSidecarPath = Normalize-ProcessPath $sidecarPath
$beforeSidecars = @(Get-RuntimeSidecars $normalizedSidecarPath)
$beforePids = @($beforeSidecars | ForEach-Object { [int]$_.ProcessId })
$si = [Diagnostics.ProcessStartInfo]::new()
$si.FileName = $exePath
$si.WorkingDirectory = $work
$si.UseShellExecute = $false
$si.Environment['IOT_APP_DATA_DIR_OVERRIDE'] = $data
$parent = [Diagnostics.Process]::Start($si)
try {
    Start-Sleep -Seconds 8
    if ($parent.HasExited) { throw "native executable exited early with code $($parent.ExitCode)" }
    $afterSidecars = @(Get-RuntimeSidecars $normalizedSidecarPath)
    $candidates = @($afterSidecars | Where-Object { $beforePids -notcontains [int]$_.ProcessId })
    if (-not $candidates) {
        $afterSidecars |
            Select-Object ProcessId, ParentProcessId, ExecutablePath, CreationDate, CommandLine |
            Format-List | Out-String | Write-Output
        throw "runtime sidecar was not observed at $sidecarPath"
    }
    $children = @($candidates | Where-Object { $_.ParentProcessId -eq $parent.Id })
    $observed = if ($children) { $children } else { $candidates }
    $childPids = @($observed | Select-Object -ExpandProperty ProcessId)
    $parentMatches = @($children | Select-Object -ExpandProperty ProcessId)
    "parent_pid=$($parent.Id) sidecar_path=$sidecarPath sidecar_pids=$($childPids -join ',') parent_matched_pids=$($parentMatches -join ',') data=$data"
    # Controlled crash simulation: terminate only the exact parent PID created above.
    $parent.Kill(); $parent.WaitForExit(10000) | Out-Null
    Start-Sleep -Seconds 3
    $remaining = @(Get-RuntimeSidecars $normalizedSidecarPath | Where-Object { $childPids -contains [int]$_.ProcessId })
    foreach ($process in $remaining) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    if ($remaining.Count -ne 0) {
        Start-Sleep -Milliseconds 500
        $remaining = @(Get-RuntimeSidecars $normalizedSidecarPath | Where-Object { $childPids -contains [int]$_.ProcessId })
    }
    if ($remaining.Count -ne 0) { throw "orphaned exact sidecar process(es): $($remaining.ProcessId -join ',')" }
    'parent_terminated=True sidecar_orphan=False'
} finally {
    if ($parent -and -not $parent.HasExited) { $parent.Kill(); $parent.WaitForExit(10000) | Out-Null }
    foreach ($childPid in @($childPids)) {
        $owned = @(Get-RuntimeSidecars $normalizedSidecarPath | Where-Object { [int]$_.ProcessId -eq [int]$childPid })
        foreach ($process in $owned) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    Remove-Item -LiteralPath $data -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
