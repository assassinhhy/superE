@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo closing desktop window
call :killpid "%~dp0data\run\desktop.pid"
echo closing board service
call :killpid "%~dp0data\run\board.pid"
echo asking API to shut down
curl.exe -s -m 5 -X POST http://127.0.0.1:8766/api/shutdown >nul
rem timeout.exe fails with "input redirection is not supported" when stdin
rem is not a console, and the script used to kill the process immediately.
set /a N=0
:wait
if not exist "%~dp0data\run\ingestor.pid" if not exist "%~dp0data\run\api.pid" goto clean
if !N! GEQ 30 goto force
ping 127.0.0.1 -n 2 >nul
set /a N+=1
goto wait
:force
echo graceful shutdown exceeded 30s, killing by pid
call :killpid "%~dp0data\run\ingestor.pid"
call :killpid "%~dp0data\run\api.pid"
call :killpid "%~dp0data\run\desktop.pid"
call :killpid "%~dp0data\run\board.pid"
goto end
:clean
echo superE stopped
goto end
:end
endlocal
exit /b 0

:killpid
if not exist "%~1" exit /b 0
set /p PID=<"%~1"
if "%PID%"=="" exit /b 0
taskkill /PID %PID% /F >nul 2>&1
del /f /q "%~1" >nul 2>&1
exit /b 0
