@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"
if defined SUPERE_PYTHON (
  set "PY=%SUPERE_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
if not exist "%~dp0data\logs" mkdir "%~dp0data\logs"
if not exist "%~dp0data\run" mkdir "%~dp0data\run"
rem detach uses CreateProcess with only stdin/stdout/stderr inherited, so a
rem PowerShell pipe is not held open by the service.
echo starting ingestor
"%PY%" -m supere detach ingestor
if errorlevel 1 (
  echo [WARN] failed to start ingestor
  echo [WARN] aborting startup - see data\logs\ingestor.stdout.log and data\logs\ingestor.stderr.log
  exit /b 1
)
echo starting api on 127.0.0.1:8766
"%PY%" -m supere detach api
if errorlevel 1 (
  echo [WARN] failed to start api
  echo [WARN] aborting startup - see data\logs\api.stdout.log and data\logs\api.stderr.log
  exit /b 1
)
echo waiting for api health: http://127.0.0.1:8766/api/health (max 15s)
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "[System.Net.WebRequest]::DefaultWebProxy=$null; $url='http://127.0.0.1:8766/api/health'; $deadline=(Get-Date).AddSeconds(15); while ((Get-Date) -lt $deadline) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) { exit 0 } } catch {} Start-Sleep -Milliseconds 500 }; exit 1" <nul
if errorlevel 1 (
  echo [WARN] api did not become healthy at http://127.0.0.1:8766/api/health within 15s
  echo [WARN] aborting startup - see data\logs\api.stdout.log and data\logs\api.stderr.log
  exit /b 1
)
echo api is ready
echo starting board on 127.0.0.1:8770
"%PY%" -m supere detach board
if errorlevel 1 (
  echo [WARN] failed to start board on 127.0.0.1:8770
  echo [WARN] aborting startup - see data\logs\board.stdout.log and data\logs\board.stderr.log
  exit /b 1
)
echo starting news on 127.0.0.1:8769
rem news.local.ps1 is not committed. It sets SUPERE_LLM_* for the Chinese summaries.
if exist "%~dp0news.local.ps1" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','%~dp0news.local.ps1' -WindowStyle Hidden" <nul
) else (
  "%PY%" -m supere detach news
)
echo starting window
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%PY%' -ArgumentList '-m','supere','desktop' -WorkingDirectory '%CD%' -WindowStyle Normal -RedirectStandardOutput '%CD%\data\logs\desktop.stdout.log' -RedirectStandardError '%CD%\data\logs\desktop.stderr.log'" <nul
echo superE started. API http://127.0.0.1:8766/api/health
echo process output: data\logs\ingestor.stdout.log , data\logs\api.stdout.log , data\logs\desktop.stdout.log
endlocal
