# Desktop shortcut target. If superE is already running it only opens the
# window (and starts board/news if they are missing), so a second click does
# not start duplicate services or overwrite their pid files.
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$pyw = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $pyw)) { $pyw = $py }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:NO_PROXY = '127.0.0.1,localhost'
function Test-Port($p) { [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) }
if (-not (Test-Port 8766)) {
    & cmd.exe /c (Join-Path $PSScriptRoot '启动superE.bat')
    exit
}
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot 'data\logs') | Out-Null
if (-not (Test-Port 8770)) { & $py -m supere detach board | Out-Null }
if (-not (Test-Port 8769)) {
    $local = Join-Path $PSScriptRoot 'news.local.ps1'
    if (Test-Path $local) {
        Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$local -WindowStyle Hidden
    } else {
        & $py -m supere detach news | Out-Null
    }
}
$alive = $false
$dp = Join-Path $PSScriptRoot 'data\run\desktop.pid'
if (Test-Path $dp) {
    $id = (Get-Content $dp -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($id -and (Get-Process -Id $id -ErrorAction SilentlyContinue)) { $alive = $true }
}
if (-not $alive) {
    Start-Process -FilePath $pyw -ArgumentList '-m','supere','desktop' -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput (Join-Path $PSScriptRoot 'data\logs\desktop.stdout.log') `
        -RedirectStandardError (Join-Path $PSScriptRoot 'data\logs\desktop.stderr.log')
}