@echo off
echo ============================================================
echo  Autobot Local Agent Launcher
echo ============================================================
echo.

:: Install Python dependencies
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip install failed. Is Python installed?
    pause
    exit /b 1
)

:: Install Playwright browsers (first run only)
echo [2/3] Checking Playwright browsers...
python -m playwright install chromium 2>nul
if %ERRORLEVEL% neq 0 (
    echo WARNING: Playwright browser install failed (may already be installed).
)

:: Start Autobot
echo [3/3] Starting Autobot on http://127.0.0.1:8000
echo.
echo  Dashboard:  http://127.0.0.1:8000
echo  API docs:   http://127.0.0.1:8000/docs
echo.
echo  To expose publicly (for Vercel frontend), run in another terminal:
echo    cloudflared tunnel --url http://localhost:8000
echo  Then paste the tunnel URL into the Vercel deploy's VITE_API_BASE env var.
echo.
python -m autobot
pause
