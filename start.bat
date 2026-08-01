@echo off
echo ============================================================
echo  Autobot Local Agent Launcher (Windows)
echo ============================================================
echo.

:: 1. Setup Python Virtual Environment
SET PYTHON_CMD=python
py -0 >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PYTHON_CMD=py
)

IF NOT EXIST venv (
    echo [1/4] Creating Python virtual environment...
    %PYTHON_CMD% -3.13 -m venv venv 2>nul || %PYTHON_CMD% -3 -m venv venv 2>nul || %PYTHON_CMD% -m venv venv 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment.
        echo Please ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
) ELSE (
    echo [1/4] Virtual environment already exists.
)

echo Activating virtual environment...
IF EXIST venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) ELSE (
    echo ERROR: Virtual environment exists but activation script not found.
    echo Please delete the 'venv' folder and try again.
    pause
    exit /b 1
)

echo [2/4] Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip install failed.
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
echo [4/4] Launching Autobot...
echo.
echo ============================================================
echo  Starting Server...
echo  Access Dashboard at: http://127.0.0.1:8000
echo  Leave this window open to keep the server running.
echo ============================================================
echo.

set PYTHONPATH=%~dp0
:: Check if frontend is built
if not exist "frontend\dist" (
    echo [!] Frontend build not found. Running in Dev Mode (Separate Servers)...
    start "Autobot Backend" cmd /c "call venv\Scripts\activate.bat && python -m autobot.main"
    cd frontend
    start "Autobot Frontend" cmd /c "npm run dev"
    cd ..
) else (
    echo [!] Frontend build found. Running in Unified Mode...
    call venv\Scripts\activate.bat && python -m autobot.cli --server
)

pause
