# AI Customer Intelligence Platform - Windows PowerShell Start Script

Write-Host "🤖 AI Customer Intelligence Platform - Quick Start" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check Python
Write-Host "`n📍 Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python version: $pythonVersion" -ForegroundColor Green

# Check if in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment is active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠️  No virtual environment detected" -ForegroundColor Yellow
    Write-Host "Recommended: Create and activate a venv first" -ForegroundColor Yellow
    $createVenv = Read-Host "Create virtual environment now? (y/n)"
    if ($createVenv -eq 'y') {
        Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        .\venv\Scripts\Activate.ps1
        Write-Host "✓ Virtual environment created and activated" -ForegroundColor Green
    }
}

# Install dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Check .env file
if (-not (Test-Path .env)) {
    Write-Host "`n⚙️  Creating .env file..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠️  IMPORTANT: Please edit .env and add your OpenRouter API key!" -ForegroundColor Red
    Write-Host "Get your key at: https://openrouter.ai/" -ForegroundColor Yellow
    $continue = Read-Host "Press Enter after adding your API key, or 'skip' to continue anyway"
    if ($continue -eq 'skip') {
        Write-Host "⚠️  Continuing without API key - AI features will not work" -ForegroundColor Yellow
    }
}

# Train models
Write-Host "`n🤖 Training ML models..." -ForegroundColor Yellow
Write-Host "This may take 1-3 minutes..." -ForegroundColor Cyan
python backend.py --train

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Training failed. Check error messages above." -ForegroundColor Red
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  1. Missing CSV files in current directory" -ForegroundColor Yellow
    Write-Host "  2. Insufficient memory" -ForegroundColor Yellow
    Write-Host "  3. Missing dependencies" -ForegroundColor Yellow
    exit 1
}

# Check ports
Write-Host "`n🔍 Checking ports..." -ForegroundColor Yellow
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
$port8501 = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue

if ($port8000) {
    Write-Host "⚠️  Port 8000 is already in use" -ForegroundColor Yellow
}
if ($port8501) {
    Write-Host "⚠️  Port 8501 is already in use" -ForegroundColor Yellow
}

# Start API server
Write-Host "`n🚀 Starting API server on port 8000..." -ForegroundColor Yellow
$apiProcess = Start-Process python -ArgumentList "backend.py --serve" -PassThru -WindowStyle Hidden
Write-Host "✓ API server started (PID: $($apiProcess.Id))" -ForegroundColor Green

# Wait for API to be ready
Write-Host "Waiting for API to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 3

# Test API
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ API server is running and healthy" -ForegroundColor Green
} catch {
    Write-Host "⚠️  API server may not be ready yet" -ForegroundColor Yellow
}

# Start Streamlit dashboard
Write-Host "`n📊 Starting Streamlit dashboard on port 8501..." -ForegroundColor Yellow
$dashboardProcess = Start-Process streamlit -ArgumentList "run dashboard.py" -PassThru -WindowStyle Hidden
Write-Host "✓ Dashboard started (PID: $($dashboardProcess.Id))" -ForegroundColor Green

Start-Sleep -Seconds 2

# Summary
Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "✅ AI Customer Intelligence Platform is running!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Dashboard:  http://localhost:8501" -ForegroundColor Cyan
Write-Host "🔌 API Docs:   http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "❤️  Health:    http://localhost:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Process IDs:" -ForegroundColor Yellow
Write-Host "  API:       $($apiProcess.Id)" -ForegroundColor White
Write-Host "  Dashboard: $($dashboardProcess.Id)" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  To stop all services, close this window or press Ctrl+C" -ForegroundColor Yellow
Write-Host ""

# Open browser
$openBrowser = Read-Host "Open dashboard in browser? (y/n)"
if ($openBrowser -eq 'y') {
    Start-Process "http://localhost:8501"
}

# Keep window open
Write-Host "`nPress Ctrl+C to stop all services..." -ForegroundColor Yellow
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host "`n🛑 Stopping services..." -ForegroundColor Yellow
    Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $dashboardProcess.Id -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Services stopped" -ForegroundColor Green
}
