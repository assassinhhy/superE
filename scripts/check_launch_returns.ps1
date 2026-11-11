# Launch 启动superE.bat through a PowerShell pipe and require it to return.
# Windows PowerShell 5.1. ASCII only. This starts superE, then stops it.
# Run from anywhere:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_launch_returns.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $root "启动superE.bat"
$stop = Join-Path $root "停止superE.bat"
if (-not (Test-Path -LiteralPath $bat)) {
    Write-Output "FAIL start script not found: $bat"
    exit 1
}

$job = Start-Job -ArgumentList $bat -ScriptBlock {
    param($startBat)
    Set-Location (Split-Path -Parent $startBat)
    "probe" | & cmd.exe /c $startBat | Out-Null
}

$finished = Wait-Job $job -Timeout 30
if (-not $finished) {
    Stop-Job $job -Force
    Remove-Job $job -Force
    Write-Output "FAIL launch did not return within 30s"
    if (Test-Path -LiteralPath $stop) {
        & cmd.exe /c $stop | Out-Null
    }
    exit 1
}

Receive-Job $job | Out-Null
Remove-Job $job -Force
Write-Output "PASS launch returned within 30s"
if (Test-Path -LiteralPath $stop) {
    & cmd.exe /c $stop | Out-Null
}
exit 0
