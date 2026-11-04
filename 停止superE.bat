@echo off
setlocal
cd /d "%~dp0"
echo asking API to shut down
curl.exe -s -m 5 -X POST http://127.0.0.1:8766/api/shutdown
if errorlevel 1 echo API shutdown request failed, will fall back to PID files
timeout /t 3 /nobreak >nul
call :killpid "%~dp0data\run\ingestor.pid"
call :killpid "%~dp0data\run\api.pid"
echo superE stop requested
endlocal
exit /b 0

:killpid
if not exist "%~1" exit /b 0
set /p PID=<"%~1"
if "%PID%"=="" exit /b 0
taskkill /PID %PID% /F >nul 2>&1
del /f /q "%~1" >nul 2>&1
exit /b 0
