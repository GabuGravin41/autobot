#!/bin/bash
# ============================================================================
# Autobot Desktop App Builder
# Packages the Python backend + pre-built frontend into a single executable
# ============================================================================
set -e

echo "=================================="
echo " Building Autobot Desktop App     "
echo "=================================="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 1. Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No venv found. Run start.sh first to set up the environment."
    exit 1
fi

# 2. Install PyInstaller if needed
pip install pyinstaller 2>/dev/null || true

# 3. Build frontend
echo ">> [1/3] Building frontend..."
cd frontend
npm install --silent
npm run build
cd "$PROJECT_DIR"

# 4. Verify frontend dist exists
if [ ! -d "frontend/dist" ]; then
    echo "Error: frontend/dist not found. Frontend build failed."
    exit 1
fi

# 5. Package with PyInstaller
echo ">> [2/3] Packaging with PyInstaller..."
pyinstaller \
    --name "Autobot" \
    --onedir \
    --noconfirm \
    --add-data "frontend/dist:frontend/dist" \
    --add-data ".env:.env_default" \
    --add-data "autobot/prompts:autobot/prompts" \
    --hidden-import "uvicorn.logging" \
    --hidden-import "uvicorn.protocols.http" \
    --hidden-import "uvicorn.protocols.http.auto" \
    --hidden-import "uvicorn.protocols.http.h11_impl" \
    --hidden-import "uvicorn.protocols.websockets" \
    --hidden-import "uvicorn.protocols.websockets.auto" \
    --hidden-import "uvicorn.protocols.websockets.websockets_impl" \
    --hidden-import "uvicorn.lifespan" \
    --hidden-import "uvicorn.lifespan.on" \
    --hidden-import "uvicorn.lifespan.off" \
    --hidden-import "fastapi" \
    --hidden-import "pydantic" \
    --hidden-import "openai" \
    --hidden-import "httpx" \
    --hidden-import "PIL" \
    --hidden-import "pyautogui" \
    --hidden-import "dotenv" \
    --collect-all "autobot" \
    autobot/main.py

echo ">> [3/3] Build complete!"
echo ""
echo "=================================="
echo " Desktop app built successfully!  "
echo " Location: dist/Autobot/          "
echo " Run: ./dist/Autobot/Autobot      "
echo "=================================="
echo ""
echo "To distribute:"
echo "  1. Copy the dist/Autobot/ folder"
echo "  2. Users need Chrome installed"
echo "  3. Users create a .env file with their API key"
echo "  4. Run ./Autobot to start"
