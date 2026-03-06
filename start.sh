#!/bin/bash
set -e

echo "=================================="
echo " Starting Autobot (Linux/macOS)  "
echo "=================================="

# 1. Setup Python Virtual Environment
if [ ! -d "venv" ]; then
    echo ">> [1/4] Creating Python virtual environment..."
    python3 -m venv venv
else
    echo ">> [1/4] Virtual environment already exists."
fi

echo ">> Activating virtual environment..."
source venv/bin/activate

echo ">> [2/4] Installing backend dependencies..."
pip install -r requirements.txt > /dev/null

# 2. Setup Node environment
echo ">> [3/4] Installing frontend dependencies..."
cd frontend
npm install > /dev/null
cd ..

# 3. Start Backend in background
echo ">> [4/4] Launching FastApi backend..."
export PYTHONPATH=$(pwd)
python -m autobot.main &
BACKEND_PID=$!

# 4. Start Frontend
echo ">> [4/4] Launching React frontend..."
cd frontend
npm run dev -- --host &
FRONTEND_PID=$!

echo "=================================="
echo " Autopilot is now RUNNING!        "
echo " Backend URL: http://0.0.0.0:8000 "
echo " Frontend URL: http://localhost:3000 "
echo " Press Ctrl+C to stop both servers."
echo "=================================="

# Catch termination signal and kill both processes
trap "echo -e '\nStopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM EXIT

wait
