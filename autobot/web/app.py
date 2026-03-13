"""
autobot/web/app.py — FastAPI application bridging the React frontend
to the new Agent architecture.

API surface:
  POST /api/agent/run      → start the new agent loop
  GET  /api/agent/status   → get status of active agent
  POST /api/agent/cancel   → cancel active agent

  GET  /api/settings       → read current settings
  POST /api/settings       → update .env-based settings
  GET  /api/runs           → historical runs
  DELETE /api/runs         → format/delete all historical runs

  WS   /ws/logs            → real-time log streaming
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Set

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import base64

from ..agent.runner import AgentRunner
from ..computer.anti_sleep import anti_sleep


logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

_agent_runner: AgentRunner | None = None
_agent_status: str = "idle"  # idle | running | done | failed | cancelled
_active_run_id: str | None = None
_run_log: list[str] = []          # timestamped lines, cleared per run
_ws_clients: set[WebSocket] = set()
_event_loop: asyncio.AbstractEventLoop | None = None
_agent_lock = threading.Lock()
_log_seq: int = 0                 # monotonic counter for dedup by frontend


# ── Logging + WebSocket broadcast ────────────────────────────────────────────

def _log(msg: str) -> None:
    global _log_seq
    ts = datetime.now().strftime("%H:%M:%S")
    # Prefix with a unique sequence number so the frontend can deduplicate
    # when multiple WebSocket connections are open (e.g. React StrictMode).
    _log_seq += 1
    line = f"[{ts}] {msg}"
    _run_log.append(line)
    # Send seq|line so clients can drop duplicates by tracking the highest seq seen
    payload = f"{_log_seq}|{line}"
    if _event_loop and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _event_loop)


async def _broadcast(msg: str) -> None:
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    _log("Autobot backend starting...")
    from ..agent.scheduler import scheduler
    scheduler.start()
    yield
    _log("Autobot backend shutting down.")
    if _agent_runner:
        _agent_runner.cancel()


app = FastAPI(title="Autobot API", version="1.0.0", lifespan=lifespan)

# Allow Vite dev server
_extra_origins = [o.strip() for o in os.getenv("AUTOBOT_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8000", "http://127.0.0.1:8000",
        *_extra_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static Files (Frontend) ──────────────────────────────────────────────────
# NOTE: The actual mount() call is placed at the BOTTOM of this file, AFTER all
# API and WebSocket routes are declared.  Mounting at "/" here would shadow every
# /api/* endpoint and cause StaticFiles to intercept WebSocket upgrade requests
# (resulting in: AssertionError: assert scope["type"] == "http").
_frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if not _frontend_path.exists():
    _frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "build"


# ── Pydantic Request Models ───────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    goal: str
    max_steps: int = 25
    use_vision: bool = True


class AntiSleepRequest(BaseModel):
    enabled: bool


class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    browser_mode: str | None = None


# ── Agent Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/agent/run")
def start_agent_run(req: AgentRunRequest):
    """
    Start the AgentRunner (CDP browser + DOM intelligence).
    """
    global _agent_runner, _agent_status, _active_run_id, _run_log

    # Use millisecond precision to ensure run_id is absolutely unique even if multiple requests arrive near-simultaneously
    run_id = f"agent_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    
    with _agent_lock:
        if _agent_status == "running":
            raise HTTPException(status_code=409, detail="A run is already in progress.")
        
        _active_run_id = run_id
        _agent_status = "running"
        _run_log.clear()
        _agent_runner = AgentRunner.from_env(log_callback=_log)

    def _run_in_thread():
        global _agent_status
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_agent_runner.run(goal=req.goal, max_steps=req.max_steps))
            with _agent_lock:
                _agent_status = "done"
            # Dump history into runs folder for later retrieval
            _save_run_history(run_id, req.goal, True, result)
        except Exception as ex:
            with _agent_lock:
                _agent_status = "failed"
            _save_run_history(run_id, req.goal, False, str(ex))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    threading.Thread(target=_run_in_thread, daemon=True, name=f"agent-{run_id}").start()
    return {"run_id": run_id, "status": "started", "goal": req.goal}


@app.get("/api/agent/status")
@app.get("/api/status")  # Backwards compat for old dashboard
def get_agent_status():
    """Return the status of the new agent loop."""
    runner_status: dict[str, Any] = {}
    if _agent_runner:
        try:
            runner_status = _agent_runner.get_status()
        except Exception:
            pass
            
    # Include some backwards compat fields so the old React UI doesn't crash completely
    return {
        "status": "ok",
        "run_status": _agent_status,
        "agent_status": _agent_status,
        "run_id": _active_run_id,
        "active_run_id": _active_run_id,
        "browser": {
            "active": _agent_status == "running",
            "mode": "cdp",
        },
        "anti_sleep_enabled": anti_sleep.enabled,
        **runner_status,
    }

@app.get("/api/browser/screenshot")
async def get_browser_screenshot():
    """Returns the current background browser screenshot as an image."""
    try:
        if _agent_runner and _agent_runner.last_screenshot_path:
            p = Path(_agent_runner.last_screenshot_path)
            if p.exists():
                # Stream the file if possible, or just read it
                return Response(content=p.read_bytes(), media_type="image/png")
    except Exception as e:
        logger.debug(f"Failed to serve screenshot: {e}")
    
    return Response(content=b"", media_type="image/png")


@app.post("/api/agent/cancel")
@app.post("/api/run/{run_id}/cancel")  # Backwards compat
def cancel_agent(run_id: str = ""):
    """Cancel the running agent task."""
    global _agent_status
    if _agent_status != "running":
        raise HTTPException(status_code=400, detail=f"No agent run active.")
    if _agent_runner:
        _agent_runner.cancel()
    _agent_status = "cancelled"
    if _active_run_id:
        _save_run_history(_active_run_id, "Cancelled", False, "Cancelled by user")
    _log("⚠️ Agent run cancelled.")
    return {"status": "cancelled"}


def _save_run_history(run_id: str, goal: str, success: bool, result: str):
    """Save run details so they show up in historical runs."""
    try:
        runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        hist = {
            "plan_name": goal[:50],
            "description": goal,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "result": result,
            "completed_steps": _agent_runner.current_step if _agent_runner else 0,
            "total_steps": _agent_runner.max_steps if _agent_runner else 0,
        }
        (run_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
        (run_dir / "console.log").write_text("\n".join(_run_log), encoding="utf-8")
    except Exception as e:
        _log(f"Failed to save run history: {e}")


# ── Settings & Runs (Utility endpoints) ───────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return {
        "llm_provider": os.getenv("AUTOBOT_LLM_PROVIDER", "openrouter"),
        "llm_model": os.getenv("AUTOBOT_LLM_MODEL", ""),
        "browser_mode": "human",
        "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_google_key": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "llm_enabled": True,
    }


@app.post("/api/settings")
def update_settings(req: SettingsUpdate):
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    updates: dict[str, str] = {}
    if req.llm_provider is not None:
        updates["AUTOBOT_LLM_PROVIDER"] = req.llm_provider
    if req.llm_model is not None:
        updates["AUTOBOT_LLM_MODEL"] = req.llm_model
    if req.openrouter_api_key:
        updates["OPENROUTER_API_KEY"] = req.openrouter_api_key
        if not req.llm_provider:
            updates["AUTOBOT_LLM_PROVIDER"] = "openrouter"
    if req.openai_api_key:
        updates["OPENAI_API_KEY"] = req.openai_api_key
    if req.google_api_key:
        updates["GOOGLE_API_KEY"] = req.google_api_key
        if not req.llm_provider:
            updates["AUTOBOT_LLM_PROVIDER"] = "google"

    if updates:
        for key, val in updates.items():
            os.environ[key] = val
            
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
            updated_keys: set[str] = set()
            new_lines = []
            for line in lines:
                k = line.split("=", 1)[0].strip()
                if k in updates:
                    new_lines.append(f"{k}={updates[k]}")
                    updated_keys.add(k)
                else:
                    new_lines.append(line)
            for key, val in updates.items():
                if key not in updated_keys:
                    new_lines.append(f"{key}={val}")
            env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return {"status": "updated", "keys_changed": list(updates.keys())}


@app.get("/api/runs")
def get_runs():
    runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
    runs = []
    if runs_root.exists():
        for run_dir in sorted(runs_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            history_file = run_dir / "history.json"
            if history_file.exists():
                try:
                    data = json.loads(history_file.read_text(encoding="utf-8"))
                    data["id"] = run_dir.name
                    data["planName"] = data.get("plan_name", "unnamed")
                    data["timestamp"] = data.get("started_at", "unknown")
                    data["status"] = "success" if data.get("success") else "failed"
                    data["stepsCompleted"] = data.get("completed_steps", 0)
                    data["totalSteps"] = data.get("total_steps", 0)
                    data["progress"] = int((data["stepsCompleted"] / max(1, data["totalSteps"])) * 100)
                    
                    console_log = run_dir / "console.log"
                    if console_log.exists():
                        lines = console_log.read_text(encoding="utf-8").splitlines()
                        data["logs"] = lines[-10:] if lines else []
                    else:
                        data["logs"] = []

                    runs.append(data)
                except Exception:
                    pass
    return {"runs": runs[:50]}


@app.delete("/api/runs")
def clear_all_runs():
    runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
    if runs_root.exists():
        import shutil
        for item in runs_root.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                except Exception:
                    pass
    return {"status": "cleared"}


@app.get("/api/run/{run_id}")
def get_run(run_id: str):
    if run_id == _active_run_id:
        return {
            "id": run_id,
            "planName": "Active Run",
            "status": _agent_status,
            "stepsCompleted": _agent_runner.current_step if _agent_runner else 0,
            "totalSteps": _agent_runner.max_steps if _agent_runner else 0,
            "logs": list(_run_log),
            "active": True,
        }
    runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
    run_dir = runs_root / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    history_file = run_dir / "history.json"
    if not history_file.exists():
        raise HTTPException(status_code=404, detail="History file not found")
    
    data = json.loads(history_file.read_text(encoding="utf-8"))
    data["id"] = run_id
    console_log = run_dir / "console.log"
    if console_log.exists():
        data["logs"] = console_log.read_text(encoding="utf-8").splitlines()
    
    data["planName"] = data.get("plan_name", "historical_run")
    data["timestamp"] = data.get("started_at", "unknown")
    data["status"] = "success" if data.get("success") else "failed"
    data["stepsCompleted"] = data.get("completed_steps", 0)
    data["totalSteps"] = data.get("total_steps", 0)
    return data


# ── WebSocket: real-time log streaming ───────────────────────────────────────

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    # Replay existing log; use negative seq IDs so frontend ignores them
    # as "historical" and doesn't duplicate with live broadcasts
    for i, line in enumerate(list(_run_log)):
        try:
            await websocket.send_text(f"h{i}|{line}")
        except Exception:
            break
    try:
        while True:
            await asyncio.sleep(30)
            dead = []
            for ws in list(_ws_clients):
                try:
                    await ws.send_text("__ping__")
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _ws_clients.discard(ws)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_clients.discard(websocket)


# ── Utility Endpoints ─────────────────────────────────────────────────────────

@app.post("/api/utils/anti-sleep")
def toggle_anti_sleep(req: AntiSleepRequest):
    """Enable or disable the anti-sleep mouse mover."""
    # We use a dummy runner if none exists to access the global anti_sleep instance
    # or just use the global anti_sleep directly if we imported it
    from ..computer.anti_sleep import anti_sleep
    if req.enabled:
        anti_sleep.start()
    else:
        anti_sleep.stop()
    return {"status": "success", "enabled": anti_sleep.enabled}

@app.get("/api/utils/anti-sleep")
def get_anti_sleep_status():
    from ..computer.anti_sleep import anti_sleep
    return {"enabled": anti_sleep.enabled}


# ── Scheduler Endpoints ───────────────────────────────────────────────────────

@app.get("/api/tasks")
async def get_tasks():
    from ..agent.scheduler import scheduler
    return scheduler.get_tasks()

@app.post("/api/tasks")
async def add_task(req: AgentRunRequest):
    from ..agent.scheduler import scheduler
    task_id = await scheduler.add_task(req.goal)
    return {"status": "queued", "task_id": task_id}

@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    from ..agent.scheduler import scheduler
    await scheduler.cancel_task(task_id)
    return {"status": "cancelled", "task_id": task_id}


# ── Server configuration ──────────────────────────────────────────────────────

# Stub routes for old React components that crash if 404
@app.get("/api/workflows")
def stub_workflows(): return {"workflows": []}

@app.get("/api/adapters")
def stub_adapters(): return {"adapters": []}

@app.get("/api/human_input")
def stub_human_input(): return {"pending": False}

@app.get("/api/logs")
def get_logs(limit: int = 500):
    global _run_log
    return {"logs": _run_log[-limit:] if _run_log else []}

class ChatRequest(BaseModel):
    message: str
    state: dict = {}

@app.post("/api/chat")
def chat(req: ChatRequest):
    plan = {
        "id": f"plan_{int(time.time())}",
        "name": "Auto Task",
        "description": req.message,
        "steps": [{"action": "auto_execute", "args": {}, "description": f"Autonomously execute: {req.message}"}]
    }
    return {
        "reply": "I have created a direct automation plan for your task. Click Execute to begin.",
        "plan": plan
    }


# ── Static Files mount — MUST be last so it doesn't shadow API/WS routes ─────
# Mounting at "/" is a catch-all; it must come after every @app route and
# @app.websocket declaration to avoid intercepting /api/* and /ws/* traffic.
if _frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_path), html=True), name="static")
else:
    @app.get("/")
    async def root_fallback():
        return {"status": "backend running", "frontend": "not built (use npm run build)"}
