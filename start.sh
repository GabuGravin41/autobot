#!/bin/bash
set -e

echo "=================================="
echo "   Starting Autobot (Linux/macOS) "
echo "=================================="

# ── Pre-flight checks ──────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 not found. Please install Python 3.10+."; exit 1; }
command -v npm    >/dev/null 2>&1 || { echo "❌ npm not found. Please install Node.js 18+."; exit 1; }

# ── 1. Python virtual environment ─────────────────────────────────
if [ ! -d "venv" ]; then
    echo ">> [1/4] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo ">> [1/4] Virtual environment already exists."
fi

echo ">> Activating virtual environment..."
source venv/bin/activate

echo ">> [2/4] Installing backend dependencies..."
pip install -r requirements.txt -q

# ── 2. Frontend build ──────────────────────────────────────────────
echo ">> [3/4] Building frontend..."
cd frontend
npm install -q
npm run build
cd ..

# ── 3. Check .env ──────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "   → Edit .env and add your API key before running tasks."
    echo ""
fi

# ── 4. Launch backend ──────────────────────────────────────────────
echo ">> [4/4] Launching Autobot backend..."
export PYTHONPATH=$(pwd)
python -m autobot.main &
BACKEND_PID=$!

LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "=================================="
echo "   Autobot is now RUNNING!"
echo "   Local: http://127.0.0.1:8000"
[ -n "$LAN_IP" ] && echo "   Phone: http://$LAN_IP:8000"
echo "   Press Ctrl+C to stop."
echo "=================================="
echo ""

# Catch termination and clean up
trap "echo -e '\nStopping Autobot...'; kill $BACKEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM EXIT

wait
