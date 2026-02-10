@echo off
REM AI Customer Intelligence Platform - Windows Batch Script
REM For Command Prompt (CMD)

echo ========================================
echo AI Customer Intelligence Platform
echo Quick Start for Windows
echo ========================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

REM Check .env
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env >nul
    echo.
    echo ============================================
    echo IMPORTANT: Edit .env and add your API key!
    echo Get it from: https://openrouter.ai/
    echo ============================================
    echo.
    echo Opening .env in notepad...
    timeout /t 2 >nul
    notepad .env
    echo.
    echo After adding your API key, press any key to continue...
    pause >nul
)

REM Train models
echo ========================================
echo Training ML models...
echo This will take 1-3 minutes
echo ========================================
echo.
python backend.py --train
if errorlevel 1 (
    echo ERROR: Training failed
    echo Check the error messages above
    pause
    exit /b 1
)
echo.
echo Training completed successfully!
echo.

REM Check ports
echo Checking if ports are available...
netstat -ano | findstr :8000 >nul
if not errorlevel 1 (
    echo WARNING: Port 8000 is already in use
)
netstat -ano | findstr :8501 >nul
if not errorlevel 1 (
    echo WARNING: Port 8501 is already in use
)
echo.

REM Start services
echo ========================================
echo Starting services...
echo ========================================
echo.

echo Starting API server on port 8000...
start "AI Platform - API Server" cmd /k "venv\Scripts\activate.bat && python backend.py --serve"
timeout /t 3 >nul

echo Starting Streamlit dashboard on port 8501...
start "AI Platform - Dashboard" cmd /k "venv\Scripts\activate.bat && streamlit run dashboard.py"
timeout /t 3 >nul

echo.
echo ========================================
echo Services are starting!
echo ========================================
echo.
echo Dashboard:  http://localhost:8501
echo API Docs:   http://localhost:8000/docs
echo Health:     http://localhost:8000/health
echo.
echo Two new windows have opened:
echo   1. API Server (port 8000)
echo   2. Dashboard (port 8501)
echo.
echo Close those windows to stop the services.
echo.
echo Opening dashboard in browser in 5 seconds...
timeout /t 5 >nul
start http://localhost:8501
echo.
echo Press any key to close this window...
pause >nul
