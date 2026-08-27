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

echo [INFO] Starting Setu API backend server on http://localhost:8000...
echo [INFO] Press Ctrl+C to stop the server.
echo.

start http://localhost:8000

"%PYTHON_CMD%" -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

pause
