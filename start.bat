@echo off
echo ============================================================
echo  Autobot Local Agent Launcher (Windows)
echo ============================================================
echo.

:: 1. Setup Python Virtual Environment
IF NOT EXIST venv (
    echo [1/4] Creating Python virtual environment...
    python -m venv venv
) ELSE (
    echo [1/4] Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/4] Installing Python dependencies...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip install failed. Is Python installed?
    pause
    exit /b 1
)

:: Install Playwright browsers (first run only)
echo [3/4] Checking Playwright browsers...
python -m playwright install chromium 2>nul
if %ERRORLEVEL% neq 0 (
    echo WARNING: Playwright browser install failed (may already be installed).
)

:: Setup Node environment & Start
echo [4/4] Installing frontend dependencies and launching...
cd frontend
call npm install --silent

echo.
echo ============================================================
echo  Starting Servers...
echo  Backend: http://127.0.0.1:8000
echo  Frontend: http://localhost:3000
echo  Leave this window open to keep servers running.
echo ============================================================
echo.

set PYTHONPATH=%~dp0
start "Autobot Backend" cmd /c "call ..\venv\Scripts\activate.bat && python -m autobot.main"
start "Autobot Frontend" cmd /c "npm run dev"

cd ..
pause
