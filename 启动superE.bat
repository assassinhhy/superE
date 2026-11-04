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
rem Start-Process does not inherit the caller's pipe, so PowerShell
rem "bat | ..." does not stay blocked on the child stdout handle.
echo starting ingestor
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%PY%' -ArgumentList '-m','supere','ingestor' -WorkingDirectory '%CD%' -WindowStyle Minimized -RedirectStandardOutput '%CD%\data\logs\ingestor.stdout.log' -RedirectStandardError '%CD%\data\logs\ingestor.stderr.log'" <nul
ping 127.0.0.1 -n 3 >nul
echo starting api on 127.0.0.1:8766
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%PY%' -ArgumentList '-m','supere','api' -WorkingDirectory '%CD%' -WindowStyle Minimized -RedirectStandardOutput '%CD%\data\logs\api.stdout.log' -RedirectStandardError '%CD%\data\logs\api.stderr.log'" <nul
echo superE started. API http://127.0.0.1:8766/api/health
echo stdout logs: data\logs\ingestor.stdout.log and data\logs\api.stdout.log
endlocal
