@echo off
setlocal
cd /d "%~dp0"
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
if not exist "%~dp0data\run" mkdir "%~dp0data\run"
echo starting ingestor with %PY%
start "superE-ingestor" /MIN "%PY%" -m supere ingestor
timeout /t 2 /nobreak >nul
echo starting api on 127.0.0.1:8766
start "superE-api" /MIN "%PY%" -m supere api
echo superE started. API http://127.0.0.1:8766/api/health
endlocal
