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
) else if exist "D:\dev\python\3.13.15\python.exe" (
  set "PY=D:\dev\python\3.13.15\python.exe"
) else (
  set "PY=python"
)
if not exist "%~dp0data\logs" mkdir "%~dp0data\logs"
if not exist "%~dp0data\run" mkdir "%~dp0data\run"
rem detach uses CreateProcess with only stdin/stdout/stderr inherited, so a
rem PowerShell pipe is not held open by the service.
echo starting ingestor
"%PY%" -m supere detach ingestor
if errorlevel 1 echo failed to detach ingestor & exit /b 1
ping 127.0.0.1 -n 3 >nul
echo starting api on 127.0.0.1:8766
"%PY%" -m supere detach api
if errorlevel 1 echo failed to detach api & exit /b 1
echo superE started. API http://127.0.0.1:8766/api/health
echo process output: data\logs\ingestor.stdout.log and data\logs\api.stdout.log
endlocal
