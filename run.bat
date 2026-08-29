@echo off
title Setu — AI Scheme Navigator

echo ===================================================
echo           Setu — AI Scheme Navigator               
echo ===================================================
echo.

cd /d "%~dp0"

set PYTHON_CMD=python

if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
    echo [INFO] Using virtual environment at venv\
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    echo [INFO] Using virtual environment at .venv\
) else (
    echo [WARNING] No venv found. Falling back to system python.
)

if not exist ".env" (
    echo [WARNING] .env file not found! Make sure GROQ_API_KEY is configured.
)

if exist "voice_service\main.py" (
    echo [INFO] Starting Setu Voice Service on http://localhost:8001...
    start "Setu Voice Service" "%PYTHON_CMD%" -m uvicorn voice_service.main:app --host 127.0.0.1 --port 8001
)

echo [INFO] Starting Setu API backend server on http://localhost:8000...
echo [INFO] Press Ctrl+C to stop the server.
echo.

:: Open browser after launch command
:: Wait 5 seconds for the server to load, then open the browser
start "" cmd /c "timeout /t 5 >nul && start http://127.0.0.1:8000"

"%PYTHON_CMD%" -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

pause
