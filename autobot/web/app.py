"""
autobot/web/app.py — FastAPI application bridging the React frontend
to the new Agent architecture.

API surface:
  POST /api/agent/run         → start the new agent loop
  GET  /api/agent/status      → get status of active agent
  POST /api/agent/pause       → pause active agent (resumes from same step)
  POST /api/agent/resume      → resume a paused agent
  POST /api/agent/cancel      → cancel active agent
  POST /api/mission/leetcode  → LeetCode multi-AI solving mission
  POST /api/task/background   → non-visual background task (parallel)

  GET  /api/settings       → read current settings
  POST /api/settings       → update .env-based settings
  GET  /api/runs           → historical runs
  DELETE /api/runs         → clear all historical runs
  DELETE /api/run/{run_id} → delete a single run
  GET  /api/run/{run_id}   → get details of a single run

  GET  /api/workflows           → list all workflows (builtin + user)
  POST /api/workflows/save      → save a new user workflow
  POST /api/workflows/{id}/run  → run a workflow
  DELETE /api/workflows/{id}    → delete a user workflow

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

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})

# Allow Vite dev server + remote monitoring (Vercel, phone, etc.)
_extra_origins = [o.strip() for o in os.getenv("AUTOBOT_CORS_ORIGINS", "").split(",") if o.strip()]
_allow_all_origins = os.getenv("AUTOBOT_CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all_origins else [
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
    max_steps: int = 100   # Match AUTOBOT_MAX_STEPS default (was 25 — too low for real tasks)
    use_vision: bool = True


class AntiSleepRequest(BaseModel):
    enabled: bool


class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    vertex_api_key: str | None = None
    xai_api_key: str | None = None
    browser_mode: str | None = None
    approval_mode: str | None = None   # "strict" | "balanced" | "trusted"


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


class LeetCodeRunRequest(BaseModel):
    num_problems: int = 5
    language: str = "python3"


@app.post("/api/mission/leetcode")
def start_leetcode_mission(req: LeetCodeRunRequest):
    """
    Start the LeetCode multi-AI solving mission.

    Opens LeetCode + Claude/Grok/DeepSeek tabs and solves unsolved problems
    by consulting each AI, picking the best solution, and submitting.
    Tracks accuracy across all attempted problems.
    """
    global _agent_runner, _agent_status, _active_run_id, _run_log

    run_id = f"leetcode_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

    with _agent_lock:
        if _agent_status == "running":
            raise HTTPException(status_code=409, detail="A run is already in progress.")
        _active_run_id = run_id
        _agent_status = "running"
        _run_log.clear()

    from ..missions.leetcode import LeetCodeMission
    lc_mission = LeetCodeMission.from_env(
        num_problems=req.num_problems,
        language=req.language,
        log_callback=_log,
    )
    _agent_runner = lc_mission.agent_runner

    def _run_in_thread():
        global _agent_status
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(lc_mission.run())
            with _agent_lock:
                _agent_status = "done"
            _save_run_history(run_id, f"LeetCode: {req.num_problems} problems", True, result)
        except Exception as ex:
            with _agent_lock:
                _agent_status = "failed"
            _save_run_history(run_id, f"LeetCode: {req.num_problems} problems", False, str(ex))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    threading.Thread(target=_run_in_thread, daemon=True, name=f"leetcode-{run_id}").start()
    return {
        "run_id": run_id,
        "status": "started",
        "goal": f"Solve {req.num_problems} LeetCode problems in {req.language}",
        "mode": "leetcode_mission",
    }


class MissionRunRequest(BaseModel):
    goal: str


@app.post("/api/mission/run")
def start_mission_run(req: MissionRunRequest):
    """
    Start a multi-objective mission. The MissionAgent plans objectives and executes each one.
    Best for complex tasks: Kaggle competitions, research workflows, multi-app coding.
    """
    global _agent_runner, _agent_status, _active_run_id, _run_log

    run_id = f"mission_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

    with _agent_lock:
        if _agent_status == "running":
            raise HTTPException(status_code=409, detail="A run is already in progress.")
        _active_run_id = run_id
        _agent_status = "running"
        _run_log.clear()
        _agent_runner = AgentRunner.from_env(log_callback=_log)

    def _run_mission_in_thread():
        global _agent_status
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_agent_runner.run_mission(goal=req.goal))
            with _agent_lock:
                _agent_status = "done"
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

    threading.Thread(target=_run_mission_in_thread, daemon=True, name=f"mission-{run_id}").start()
    return {"run_id": run_id, "status": "started", "goal": req.goal, "mode": "mission"}


# ── Orchestrated run ──────────────────────────────────────────────────────────

@app.post("/api/orchestrate")
def start_orchestrated_run(req: MissionRunRequest):
    """
    Start an orchestrated multi-agent run.

    Automatically routes to the right specialist agents based on task type.
    Simple tasks → single AgentLoop (no overhead).
    Complex tasks → Orchestrator decomposes into sub-tasks for specialists.
    """
    global _agent_runner, _agent_status, _active_run_id, _run_log

    run_id = f"orchestrated_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

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
            result = loop.run_until_complete(
                _agent_runner.run_orchestrated(goal=req.goal)
            )
            with _agent_lock:
                _agent_status = "done"
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

    threading.Thread(target=_run_in_thread, daemon=True, name=f"orchestrated-{run_id}").start()
    return {"run_id": run_id, "status": "started", "goal": req.goal, "mode": "orchestrated"}


@app.get("/api/orchestrate/plan")
def get_orchestration_plan():
    """Return the current orchestration plan status (tasks, progress, etc.)."""
    if _agent_runner is None:
        return {"plan": None}
    try:
        # The runner's orchestrated run stores the orchestrator as _orchestrator
        orchestrator = getattr(_agent_runner, "_orchestrator", None)
        if orchestrator is None:
            # Try the run_orchestrated's orchestrator (set during run)
            return {"plan": None}
        return {"plan": orchestrator.get_plan_status()}
    except Exception as e:
        return {"plan": None, "error": str(e)}


# ── RL stats endpoint ─────────────────────────────────────────────────────────

@app.get("/api/learning/stats")
def get_learning_stats():
    """
    Return RL pipeline stats — experiences accumulated, learned contexts, etc.
    Useful for understanding how much the agent has learned across runs.
    """
    try:
        from ..learning.rl_controller import rl_controller
        from ..learning.policy_memory import policy_memory, wait_duration_memory
        from ..memory.store import memory_store
        stats = rl_controller.get_stats()
        policy_summary = policy_memory.summary()
        mem_stats = memory_store.stats()
        wait_stats = wait_duration_memory.summary()
        return {
            "rl_enabled": stats.get("enabled", False),
            "total_experiences": stats.get("total_experiences", 0),
            "learned_contexts": policy_summary.get("contexts", 0),
            "total_policy_observations": policy_summary.get("total_observations", 0),
            "current_run_steps": stats.get("step_counter", 0),
            "run_id": stats.get("run_id", ""),
            "memory_entries": mem_stats.get("total", 0),
            "memory_hits": mem_stats.get("total_hits", 0),
            "memory_high_value": mem_stats.get("high_value_entries", 0),
            "wait_url_patterns": wait_stats.get("url_patterns", 0),
            "wait_observations": wait_stats.get("total_observations", 0),
        }
    except Exception as e:
        return {"rl_enabled": False, "error": str(e)}


@app.get("/api/memory/entries")
def get_memory_entries():
    """Return all stored memory entries (newest first, up to 200)."""
    try:
        from ..memory.store import memory_store
        entries = memory_store.all_entries()[:200]
        return {"entries": [{"key": k, "value": v} for k, v in entries], "total": len(memory_store)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory/prune")
def prune_memory():
    """Prune stale or zero-hit memory entries."""
    try:
        from ..memory.store import memory_store
        removed = memory_store.prune(max_age_days=60, max_entries=500)
        stats = memory_store.stats()
        return {"removed": removed, "remaining": stats["total"], "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory/entry/{key}")
def delete_memory_entry(key: str):
    """Delete a specific memory entry by key."""
    try:
        from ..memory.store import memory_store
        memory_store.forget(key)
        return {"deleted": key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_human_approval_pending() -> dict | None:
    """Return pending HumanGate approval request for the frontend."""
    try:
        from ..agent.human_gate import get_pending
        return get_pending()
    except Exception:
        return None


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
            "url": runner_status.get("browser_url", ""),
        },
        "anti_sleep_enabled": anti_sleep.enabled,
        "auth_notification": runner_status.get("auth_notification"),
        "human_approval_pending": _get_human_approval_pending(),
        **{k: v for k, v in runner_status.items() if k != "auth_notification"},
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


class BackgroundRunRequest(BaseModel):
    goal: str
    max_steps: int = 50


@app.post("/api/task/background")
def start_background_task(req: BackgroundRunRequest):
    """
    Start a non-visual background task that runs in parallel with any active visual agent.

    Background tasks can: run terminal commands, process files, call APIs, monitor processes.
    They cannot: click, type, take screenshots, or acquire ScreenLock.

    Example goals:
      - "Run python train.py and wait for it to finish, then report final accuracy"
      - "Download the Kaggle dataset using the API and extract it to ~/data/"
      - "Monitor the running training job and alert me when loss < 0.01"
    """
    from ..agent.background_runner import BackgroundTaskRunner
    run_id = f"bg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

    bg_runner = BackgroundTaskRunner.from_env(
        log_callback=_log,
        task_id=run_id,
    )

    def _run_bg():
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(bg_runner.run(goal=req.goal))
            _save_run_history(run_id, req.goal, True, result)
        except Exception as ex:
            _save_run_history(run_id, req.goal, False, str(ex))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    import threading
    threading.Thread(target=_run_bg, daemon=True, name=f"bg-{run_id}").start()
    return {"run_id": run_id, "status": "started", "goal": req.goal, "mode": "background"}


@app.post("/api/agent/pause")
def pause_agent():
    """Pause the active agent run (it will idle after the current step)."""
    if not _agent_runner or _agent_status not in ("running",):
        raise HTTPException(status_code=400, detail="No active agent run to pause.")
    _agent_runner.pause()
    return {"status": "paused"}


@app.post("/api/agent/resume")
def resume_agent():
    """Resume a paused agent run."""
    if not _agent_runner or _agent_status not in ("paused", "running"):
        raise HTTPException(status_code=400, detail="No paused agent run to resume.")
    _agent_runner.resume()
    return {"status": "running"}


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
    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "openrouter")
    has_user_key = bool(
        os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or
        os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or
        os.getenv("XAI_API_KEY") or os.getenv("VERTEX_API_KEY")
    )
    using_default_key = bool(os.getenv("AUTOBOT_DEFAULT_API_KEY")) and not has_user_key
    return {
        "llm_provider": provider,
        "llm_model": os.getenv("AUTOBOT_LLM_MODEL", ""),
        "browser_mode": "human",
        "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_google_key": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "has_xai_key": bool(os.getenv("XAI_API_KEY")),
        "has_vertex_key": bool(os.getenv("VERTEX_API_KEY")),
        "llm_enabled": True,
        "cors_allow_all": os.getenv("AUTOBOT_CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes"),
        "approval_mode": os.getenv("AUTOBOT_APPROVAL_MODE", "balanced"),
        "using_default_key": using_default_key,
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
    if req.vertex_api_key:
        updates["VERTEX_API_KEY"] = req.vertex_api_key
        if not req.llm_provider:
            updates["AUTOBOT_LLM_PROVIDER"] = "vertex"
    if req.xai_api_key:
        updates["XAI_API_KEY"] = req.xai_api_key
        if not req.llm_provider:
            updates["AUTOBOT_LLM_PROVIDER"] = "xai"
    if req.approval_mode and req.approval_mode in ("strict", "balanced", "trusted"):
        updates["AUTOBOT_APPROVAL_MODE"] = req.approval_mode

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
                    data["goal"] = data.get("description", "")
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


@app.delete("/api/run/{run_id}")
def delete_run(run_id: str):
    """Delete a single run folder by ID."""
    import shutil
    runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
    run_dir = runs_root / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    try:
        shutil.rmtree(run_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete run: {e}")
    return {"status": "deleted", "run_id": run_id}


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

class AddTaskRequest(BaseModel):
    goal: str
    priority: int = 1
    run_at: float | None = None   # epoch timestamp; None = ASAP


@app.get("/api/tasks")
async def get_tasks():
    """List all tasks (queued, running, done, failed, cancelled)."""
    from ..agent.scheduler import scheduler
    return {"tasks": scheduler.get_all_tasks()}


@app.post("/api/tasks")
async def add_task(req: AddTaskRequest):
    """Add a task to the queue. Returns task_id."""
    from ..agent.scheduler import scheduler
    task_id = await scheduler.add_task(
        goal=req.goal,
        priority=req.priority,
        run_at=req.run_at,
    )
    return {"status": "queued", "task_id": task_id}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get detailed status for a single task."""
    from ..agent.scheduler import scheduler
    task = scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.summary()


@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a queued or running task."""
    from ..agent.scheduler import scheduler
    found = await scheduler.cancel_task(task_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"status": "cancelled", "task_id": task_id}


@app.get("/api/tasks/{task_id}/logs")
async def get_task_logs(task_id: str, since: int = 0):
    """Stream log lines for a task (poll with ?since=N to get incremental lines)."""
    from ..agent.scheduler import scheduler
    lines = scheduler.get_logs(task_id, since_line=since)
    return {"task_id": task_id, "since": since, "lines": lines, "total": since + len(lines)}


@app.get("/api/screen-lock")
def get_screen_lock_status():
    """Current screen lock holder — useful for multi-task dashboard."""
    from ..agent.resource_manager import screen_lock
    return screen_lock.get_status()


@app.get("/api/schedule/status")
async def get_schedule_status():
    """
    Full scheduling dashboard:
    - How many concurrent slots are in use vs available
    - ScreenLock holder + waiting tasks
    - Running / queued / paused counts
    """
    from ..agent.scheduler import scheduler
    return scheduler.get_schedule_status()


@app.post("/api/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """Pause a queued task so the scheduler skips it until resumed."""
    from ..agent.scheduler import scheduler
    ok = await scheduler.pause_task(task_id)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} could not be paused (not in queued state)."
        )
    return {"status": "paused", "task_id": task_id}


@app.post("/api/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    """Resume a paused task — re-queues it so the scheduler will pick it up."""
    from ..agent.scheduler import scheduler
    ok = await scheduler.resume_task(task_id)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} could not be resumed (not in paused state)."
        )
    return {"status": "queued", "task_id": task_id}


@app.patch("/api/tasks/{task_id}/priority")
async def set_task_priority(task_id: str, priority: int):
    """Change the priority of a queued/paused task (higher = runs first)."""
    from ..agent.scheduler import scheduler
    ok = await scheduler.reprioritize_task(task_id, priority)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is already running or not found."
        )
    return {"status": "updated", "task_id": task_id, "priority": priority}


# ── Server configuration ──────────────────────────────────────────────────────

# ── Built-in Workflow Templates ───────────────────────────────────────────────

_BUILTIN_WORKFLOWS = [
    {
        "id": "web_research",
        "name": "Web Research",
        "description": "Research a topic using AI chatbots (Grok, ChatGPT, etc.) and compile findings.",
        "topic_label": "Research topic",
    },
    {
        "id": "latex_paper",
        "name": "LaTeX Paper Generator",
        "description": "Research a topic, generate a LaTeX paper, open Overleaf, create a blank project, and paste the paper.",
        "topic_label": "Paper topic",
    },
    {
        "id": "email_summary",
        "name": "Email Summary",
        "description": "Open Gmail, scan recent emails, and compile a summary of important items.",
        "topic_label": "",
    },
    {
        "id": "social_post",
        "name": "Social Media Post",
        "description": "Create and post content to social media platforms.",
        "topic_label": "Post content/topic",
    },
    {
        "id": "code_review",
        "name": "Code Review Helper",
        "description": "Open GitHub, review pull requests, and summarize changes and issues.",
        "topic_label": "Repository URL or name",
    },
]

_WORKFLOWS_FILE = Path(__file__).resolve().parent.parent.parent / "workflows.json"


def _load_user_workflows() -> list[dict]:
    """Load user-saved workflows from workflows.json."""
    try:
        if _WORKFLOWS_FILE.exists():
            return json.loads(_WORKFLOWS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_user_workflows(workflows: list[dict]) -> None:
    _WORKFLOWS_FILE.write_text(json.dumps(workflows, indent=2, ensure_ascii=False), encoding="utf-8")


@app.get("/api/workflows")
def get_workflows():
    user = [dict(w, source="user") for w in _load_user_workflows()]
    builtin = [dict(w, source="builtin") for w in _BUILTIN_WORKFLOWS]
    return {"workflows": builtin + user}


class SaveWorkflowRequest(BaseModel):
    name: str
    description: str
    goal: str                       # the full goal/instruction string
    topic_label: str = ""           # optional placeholder shown in the Run card


@app.post("/api/workflows/save")
def save_workflow(req: SaveWorkflowRequest):
    """Save a goal as a reusable named workflow."""
    workflows = _load_user_workflows()
    wf_id = f"user_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    new_wf = {
        "id": wf_id,
        "name": req.name.strip(),
        "description": req.description.strip(),
        "goal": req.goal.strip(),
        "topic_label": req.topic_label.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    workflows.append(new_wf)
    _save_user_workflows(workflows)
    return {"status": "saved", "workflow": new_wf}


@app.delete("/api/workflows/{workflow_id}")
def delete_workflow(workflow_id: str):
    """Delete a user-saved workflow. Built-in workflows cannot be deleted."""
    workflows = _load_user_workflows()
    updated = [w for w in workflows if w["id"] != workflow_id]
    if len(updated) == len(workflows):
        raise HTTPException(status_code=404, detail=f"User workflow '{workflow_id}' not found.")
    _save_user_workflows(updated)
    return {"status": "deleted", "workflow_id": workflow_id}


@app.post("/api/workflows/run")
def run_workflow_endpoint(req: dict):
    """Run a built-in workflow by converting it to an agent goal."""
    global _agent_runner, _agent_status, _active_run_id, _run_log

    wf_id = req.get("workflow_id", "")
    topic = req.get("topic", "")

    wf = next((w for w in _BUILTIN_WORKFLOWS if w["id"] == wf_id), None)
    if not wf:
        wf = next((w for w in _load_user_workflows() if w["id"] == wf_id), None)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow '{wf_id}' not found")

    # User-saved workflows store the full goal; built-ins build it from description
    if wf.get("goal"):
        goal = wf["goal"]
        if topic:
            goal = f"{goal}\n\nTopic/subject: {topic}"
    else:
        goal = wf["description"]
        if topic:
            goal = f"{wf['description']} Topic/subject: {topic}"

    run_id = f"wf_{wf_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

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
            result = loop.run_until_complete(_agent_runner.run(goal=goal, max_steps=30))
            with _agent_lock:
                _agent_status = "done"
            _save_run_history(run_id, goal, True, result)
        except Exception as ex:
            with _agent_lock:
                _agent_status = "failed"
            _save_run_history(run_id, goal, False, str(ex))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    threading.Thread(target=_run_in_thread, daemon=True, name=f"wf-{run_id}").start()
    return {"run_id": run_id, "status": "started", "plan_name": wf["name"]}

@app.get("/api/adapters")
def stub_adapters(): return {"adapters": []}

@app.get("/api/human_input")
def get_human_input():
    """Return the current pending approval request (if any)."""
    from ..agent.human_gate import get_pending
    pending = get_pending()
    if pending:
        return {"pending": True, "key": pending["key"], "message": pending["message"]}
    return {"pending": False}


class HumanInputResponse(BaseModel):
    key: str
    response: str   # "allow" | "block"


@app.post("/api/human_input")
def submit_human_input(req: HumanInputResponse):
    """Respond to a pending approval request."""
    from ..agent.human_gate import respond
    found = respond(key=req.key, response=req.response)
    if not found:
        raise HTTPException(status_code=404, detail=f"No pending request for key: {req.key}")
    return {"status": "ok", "key": req.key, "response": req.response}

class OnboardingData(BaseModel):
    name: str | None = None
    kaggle_username: str | None = None
    editor: str | None = None
    ai_tools: str | None = None
    language: str | None = None


@app.post("/api/onboarding")
def submit_onboarding(data: OnboardingData):
    """Save first-run onboarding profile to persistent memory."""
    from ..memory.store import memory_store
    saved: list[str] = []
    if data.name:
        memory_store.remember("user_name", data.name)
        saved.append("name")
    if data.kaggle_username:
        memory_store.remember("kaggle_username", data.kaggle_username)
        saved.append("kaggle_username")
    if data.editor:
        memory_store.remember("preferred_editor", data.editor)
        saved.append("editor")
    if data.ai_tools:
        memory_store.remember("ai_tools", data.ai_tools)
        saved.append("ai_tools")
    if data.language:
        memory_store.remember("preferred_language", data.language)
        saved.append("language")
    # Mark onboarding as complete so the UI doesn't show it again
    memory_store.remember("onboarding_complete", "true")
    return {"status": "ok", "saved": saved}


@app.get("/api/onboarding/status")
def onboarding_status():
    """Check whether the user has completed first-run onboarding."""
    from ..memory.store import memory_store
    done = memory_store.recall("onboarding_complete", top_k=1)
    complete = any(v == "true" for _, v in done)
    return {"complete": complete}


@app.get("/api/logs")
def get_logs(limit: int = 500):
    global _run_log
    return {"logs": _run_log[-limit:] if _run_log else []}

class ChatRequest(BaseModel):
    message: str
    state: dict = {}
    history: list[dict] = []  # previous messages for multi-turn


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Multi-turn AI Planner chat. The LLM converses with the user to understand
    their task, asks clarifying questions, and only proposes a plan when ready.
    """
    try:
        return _chat_with_llm(req)
    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"The AI model is unavailable: {e}. Please check that llama-server (or your configured LLM provider) is running and try again."
        )


def _chat_with_llm(req: ChatRequest) -> dict:
    """Use the configured LLM to have a real planning conversation."""
    import asyncio

    llm_client = _create_llm_client_for_chat()
    if not llm_client:
        raise RuntimeError("No LLM client available")

    model = os.getenv("AUTOBOT_LLM_MODEL", "gpt-4o")

    system_prompt = """You are the brain of Autobot — a sovereign digital agent that controls the user's entire computer. You can see the screen, move the mouse, type on the keyboard, switch between applications, and use any software installed on the machine, exactly like a human sitting at the desk.

## Who you are

You are intelligent and conversational. You understand complex requests, ask thoughtful clarifying questions, discuss tradeoffs, and help the user think through their goals. You are like ChatGPT or Grok in your ability to understand — but unlike them, you have hands and legs. You can actually DO things on the computer.

## What you can do (your capabilities)

When executing a plan, Autobot physically controls the computer:
- **Browser:** Navigate to any website, click, type, scroll, manage tabs, read page content
- **Desktop apps:** Switch between any application (VS Code, terminal, file manager, etc.) using Alt+Tab, mouse, and keyboard
- **Clipboard:** Copy and paste between any applications
- **AI platforms as tools:** Navigate to Grok, ChatGPT, Claude, Perplexity, Google AI Studio — type questions, wait for responses, copy the answers. These are your research and thinking tools.
- **Terminal:** Run shell commands, scripts, git operations
- **File management:** Create, edit, move, upload, download files
- **Keyboard shortcuts:** Any shortcut in any application

## How you think about planning

Your superpower is that you delegate the actual intellectual work to the right tool:
- Need research? → Navigate to Grok or Perplexity, type the question, read and use the response
- Need code? → Ask ChatGPT/Claude/Grok to write it, or open VS Code with Copilot
- Need a document? → Open Google Docs or a text editor and compose it there
- Need data? → Navigate to the source website and extract it

Every step in your plan is something that physically happens on screen — navigating, clicking, typing, waiting, copying. The plan is a sequence of real computer actions that Autobot will execute autonomously.

## Conversation flow

1. **Understand the request.** Have a real conversation. If the task is complex, discuss it — ask about preferences (which AI platform? which browser? where to save results?), suggest approaches, help the user refine their goal.

2. **When you understand enough, propose a plan.** The plan is a sequence of concrete steps that Autobot will execute on the computer. Include a ```plan``` block in your response.

3. **For simple, clear requests** (like "go to youtube and search for lofi music"), propose the plan immediately — no need for back-and-forth.

## Plan format

When you're ready to propose a plan, include this block in your response:

```plan
{
  "name": "Brief plan name",
  "description": "Detailed description of the full task — this is what Autobot receives as its goal, so be specific about what to do, which sites to visit, what to type, and what the end result should look like",
  "steps": [
    {"description": "Step 1: what physically happens on screen"},
    {"description": "Step 2: what physically happens on screen"},
    {"description": "Step 3: what physically happens on screen"}
  ]
}
```

The description field is critical — it becomes the agent's goal. Make it rich and detailed: include URLs, app names, the exact queries to type, where to save results, and what success looks like.

Each step should describe a real action on the computer (navigate, click, type, wait, copy, switch app, run command). Keep steps concrete enough that someone watching the screen would see each one happen."""

    # Build conversation history
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in req.history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": req.message})

    # Call LLM synchronously (we're in a sync endpoint)
    loop = asyncio.new_event_loop()
    try:
        resp = loop.run_until_complete(
            llm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
            )
        )
    finally:
        loop.close()

    reply_text = resp.choices[0].message.content if resp.choices else ""
    if not reply_text:
        raise RuntimeError("Empty LLM response")

    # Check if the reply contains a plan block
    plan = None
    import re
    plan_match = re.search(r'```plan\s*\n(.*?)\n```', reply_text, re.DOTALL)
    if not plan_match:
        # Also try ```json blocks — small models sometimes use json instead of plan
        plan_match = re.search(r'```json\s*\n(.*?)\n```', reply_text, re.DOTALL)
        # Only accept json blocks that look like plans (have "steps" key)
        if plan_match and '"steps"' not in plan_match.group(1):
            plan_match = None

    if plan_match:
        try:
            plan_data = json.loads(plan_match.group(1))
            raw_steps = plan_data.get("steps", [])
            steps = [
                {"action": "auto_execute", "args": {}, "description": s.get("description", str(s))}
                for s in raw_steps
            ]

            # ── Validate: is this actually an executable plan? ────────────
            # Small models sometimes stuff clarifying questions or abstract
            # tasks into the plan block.  Detect and discard those.
            if steps:
                step_texts = " ".join(s["description"].lower() for s in steps)
                question_count = sum(1 for s in steps if "?" in s["description"])
                has_action_verbs = any(v in step_texts for v in (
                    "navigate", "open", "click", "type", "press", "go to",
                    "search", "scroll", "copy", "paste", "switch", "wait",
                    "run ", "download", "upload", "save", "enter",
                ))
                # Reject if: majority of steps are questions, or no action verbs at all
                is_garbage = (
                    question_count > len(steps) // 2
                    or (not has_action_verbs and len(steps) > 0)
                )
            else:
                is_garbage = True

            if not is_garbage:
                plan = {
                    "id": f"plan_{int(time.time())}",
                    "name": plan_data.get("name", "Auto Task"),
                    "description": plan_data.get("description", req.message),
                    "steps": steps,
                }
            else:
                logger.debug("Plan block discarded — steps are questions or non-actionable")

        except json.JSONDecodeError:
            pass  # Plan JSON was malformed — just return the text

        # Strip the plan/json block from the visible reply
        reply_text = reply_text[:plan_match.start()].strip()

        if plan:
            # Valid plan extracted — use default intro if LLM didn't write one
            if not reply_text:
                reply_text = "Here's the plan I've prepared for you:"
        else:
            # Plan was garbage — restore the original text as conversation
            # The LLM's "steps" were really questions or discussion, so put
            # them back into the reply as natural text
            raw_text_after = reply_text
            full_text = resp.choices[0].message.content if resp.choices else ""
            # Remove only the ```plan``` markers, keep the content as text
            reply_text = re.sub(r'```(?:plan|json)\s*\n', '', full_text)
            reply_text = reply_text.replace('```', '').strip()
            # Clean up any raw JSON artifacts for readability
            reply_text = re.sub(r'[{}\[\]"]+', '', reply_text).strip()
            if not reply_text:
                reply_text = "Hello! I'm Autobot. What would you like me to help you with today?"

    return {"reply": reply_text, "plan": plan}


def _create_llm_client_for_chat():
    """Create LLM client for the chat endpoint — mirrors _create_llm_client() in runner.py."""
    from autobot.agent.runner import _create_llm_client
    return _create_llm_client()


# ── Ngrok Tunnel ─────────────────────────────────────────────────────────────

_ngrok_process: Any = None
_ngrok_url: str | None = None


@app.post("/api/tunnel/start")
def start_tunnel():
    """Start an ngrok tunnel to expose the backend for remote monitoring."""
    global _ngrok_process, _ngrok_url
    import subprocess
    import shutil

    if _ngrok_url:
        return {"status": "already_running", "url": _ngrok_url}

    if not shutil.which("ngrok"):
        raise HTTPException(status_code=400, detail="ngrok not found. Install it: https://ngrok.com/download")

    try:
        _ngrok_process = subprocess.Popen(
            ["ngrok", "http", "8000", "--log=stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Give ngrok a moment to start, then query the local API for the URL
        import time
        time.sleep(2)
        try:
            import urllib.request
            resp = urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3)
            data = json.loads(resp.read())
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    _ngrok_url = tunnel["public_url"]
                    break
            if not _ngrok_url and data.get("tunnels"):
                _ngrok_url = data["tunnels"][0].get("public_url")
        except Exception as e:
            logger.warning(f"Could not query ngrok API: {e}")
            _ngrok_url = "starting... check http://127.0.0.1:4040"

        _log(f"🌐 Ngrok tunnel started: {_ngrok_url}")
        return {"status": "started", "url": _ngrok_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start ngrok: {e}")


@app.post("/api/tunnel/stop")
def stop_tunnel():
    """Stop the ngrok tunnel."""
    global _ngrok_process, _ngrok_url
    if _ngrok_process:
        _ngrok_process.terminate()
        _ngrok_process = None
    _ngrok_url = None
    _log("🌐 Ngrok tunnel stopped.")
    return {"status": "stopped"}


@app.get("/api/tunnel/status")
def tunnel_status():
    """Get the current tunnel status."""
    return {"active": _ngrok_url is not None, "url": _ngrok_url}


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    """Uptime check for monitoring, load balancers, and Docker HEALTHCHECK."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "agent_status": _agent_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
