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
echo stopping superE services
"%PY%" -m supere stop
if errorlevel 1 (
  echo [WARN] some services did not stop - check data\run and data\logs
  pause
)
endlocal
