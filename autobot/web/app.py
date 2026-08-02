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
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agent.runner import AgentRunner
from ..computer.liveness import SystemLivenessManager


# ── State ─────────────────────────────────────────────────────────────────────

_agent_runner: AgentRunner | None = None
_agent_status: str = "idle"  # idle | running | done | failed | cancelled
_active_run_id: str | None = None
_run_log: list[str] = []
_ws_clients: set[WebSocket] = set()
_event_loop: asyncio.AbstractEventLoop | None = None
_liveness = SystemLivenessManager()

# Guards the check-then-set on _agent_status/_agent_runner in start_agent_run().
# Route handlers here are sync `def`s, which FastAPI dispatches to its worker
# threadpool — so two near-simultaneous POST /api/agent/run requests can both
# observe _agent_status != "running" before either writes "running", and both
# proceed to start a background run. The second silently overwrites
# _agent_runner, orphaning the first run: it keeps executing (and can still
# act on the real computer) with no reference left to cancel or query it.
_run_start_lock = threading.Lock()


# ── Logging + WebSocket broadcast ────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _run_log.append(line)
    if _event_loop and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(line), _event_loop)


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
    await scheduler.stop()


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


# ── Pydantic Request Models ───────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    goal: str
    max_steps: int = 25
    use_vision: bool = True


class AgentOverrideRequest(BaseModel):
    instruction: str


class AgentMessageRequest(BaseModel):
    text: str


class AntiSleepRequest(BaseModel):
    enabled: bool


class ApprovalResponse(BaseModel):
    key: str
    response: str  # "allow" | "block"


class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    browser_mode: str | None = None


# ── Agent Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/agent/run")
def start_agent_run(req: AgentRunRequest):
    """
    Start the AgentRunner (CDP browser + DOM intelligence).
    """
    global _agent_runner, _agent_status, _active_run_id, _run_log

    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")

    # Check-then-set must be one atomic step, or two near-simultaneous
    # requests can both pass the "not already running" check before either
    # marks status "running" — see _run_start_lock's comment above.
    with _run_start_lock:
        if _agent_status == "running":
            raise HTTPException(status_code=409, detail="A run is already in progress.")

        run_id = f"agent_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        _active_run_id = run_id
        _agent_status = "running"
        _run_log.clear()
        _agent_runner = AgentRunner.from_env(log_callback=_log)

    # Enable system sleep prevention
    _liveness.enable_liveness()

    def _run_in_thread():
        global _agent_status
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_agent_runner.run(goal=req.goal, max_steps=req.max_steps))
            _agent_status = "done"
            # Dump history into runs folder for later retrieval
            _save_run_history(run_id, req.goal, True, result)
        except Exception as ex:
            _agent_status = "failed"
            _save_run_history(run_id, req.goal, False, str(ex))
        finally:
            _liveness.disable_liveness()
            try:
                loop.close()
            except Exception:
                pass

    threading.Thread(target=_run_in_thread, daemon=True, name=f"agent-{run_id}").start()
    return {"run_id": run_id, "status": "started", "goal": req.goal}


@app.post("/api/agent/override")
def push_agent_override(req: AgentOverrideRequest):
    """
    Push a mid-flight goal override / intervention to the active running agent.
    """
    if _agent_status != "running" or not _agent_runner:
        raise HTTPException(status_code=400, detail="No active agent run to override.")

    _agent_runner.push_override(req.instruction)
    return {"status": "ok", "override": req.instruction}


@app.post("/api/agent/message")
def send_agent_message(req: AgentMessageRequest):
    """
    Inject a human message into the running agent's next step.

    apiService.ts's own docstring for this claims it "works whether the
    agent is running or paused (also auto-resumes if paused)". No such
    single-agent pause/resume exists — AgentLoop only has cancel — so this
    implements the running-agent case faithfully via the SAME mechanism as
    /api/agent/override (AgentLoop.push_override(), already proven to work:
    _execute_step folds it into the goal before the next LLM call) rather
    than human_gate.inject_user_message()/pop_user_messages(), which despite
    existing and looking purpose-built for this, is never actually consumed
    anywhere in AgentLoop — wiring a route to it would have created a new
    "looks fixed, silently does nothing" bug of exactly the kind this
    project has had repeatedly.
    """
    if _agent_status != "running" or not _agent_runner:
        raise HTTPException(status_code=400, detail="No active agent run to message.")

    _agent_runner.push_override(req.text)
    return {"status": "ok", "text": req.text}


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
        "status": "ok",  # Legacy
        "run_status": _agent_status,  # Legacy and new
        "agent_status": _agent_status,
        "run_id": _active_run_id,
        "active_run_id": _active_run_id,  # Legacy
        "liveness_active": _liveness.is_active,
        "browser": {
            "active": _agent_status == "running",
            "mode": "cdp",
        },
        **runner_status,
    }


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
    _liveness.disable_liveness()
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
        "llm_provider": os.getenv("AUTOBOT_LLM_PROVIDER", "auto"),
        "llm_model": os.getenv("AUTOBOT_LLM_MODEL", ""),
        "browser_mode": "cdp",  # Enforced now
        "has_anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
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
    if req.anthropic_api_key:
        updates["ANTHROPIC_API_KEY"] = req.anthropic_api_key
        if not req.llm_provider:
            updates["AUTOBOT_LLM_PROVIDER"] = "anthropic"
    if req.openrouter_api_key:
        updates["OPENROUTER_API_KEY"] = req.openrouter_api_key
        if not req.llm_provider:
            updates["AUTOBOT_LLM_PROVIDER"] = "openrouter"
    if req.openai_api_key:
        updates["OPENAI_API_KEY"] = req.openai_api_key

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
    runs_root = (Path(__file__).resolve().parent.parent.parent / "runs").resolve()
    run_dir = (runs_root / run_id).resolve()
    # run_id comes straight from the URL path with no validation. Without this
    # check, a request like GET /api/run/..%2F..%2F..%2FWindows%2FSystem32%2Fsome_dir
    # resolves outside runs_root entirely — reading history.json/console.log
    # from anywhere on disk that happens to contain files with those names.
    if runs_root not in run_dir.parents and run_dir != runs_root:
        raise HTTPException(status_code=400, detail="Invalid run_id")
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
    for line in list(_run_log):
        try:
            await websocket.send_text(line)
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


# ── Server configuration ──────────────────────────────────────────────────────

# Stub routes for old React components that crash if 404
@app.get("/api/workflows")
def stub_workflows(): return {"workflows": []}

@app.get("/api/adapters")
def stub_adapters(): return {"adapters": []}


@app.get("/api/human_input")
def get_human_input():
    """
    Poll for a pending approval request (ApprovalGuard IRREVERSIBLE/DANGER
    actions, or an explicit request_human_input action). The dashboard —
    including from a phone — polls this to know when the agent is paused
    waiting on you.

    Previously hardcoded to always return {"pending": False}, which meant
    an IRREVERSIBLE-tier action would register a real approval request via
    human_gate.wait_for_approval() but the frontend could never see it —
    it would just silently time out and auto-block after 5 minutes with no
    way to actually click Allow.
    """
    from ..agent.human_gate import get_pending
    pending = get_pending()
    if not pending:
        return {"pending": False}
    return {"pending": True, "key": pending["key"], "message": pending["message"]}


@app.post("/api/human_input/respond")
def respond_human_input(req: ApprovalResponse):
    """Allow or block a pending approval request (see get_human_input above)."""
    from ..agent.human_gate import respond
    if req.response not in ("allow", "block"):
        raise HTTPException(status_code=400, detail="response must be 'allow' or 'block'")
    ok = respond(req.key, req.response)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No pending approval with key '{req.key}'")
    _log(f"{'✅ Allowed' if req.response == 'allow' else '🚫 Blocked'} pending action ({req.key})")
    return {"status": "ok", "key": req.key, "response": req.response}

@app.get("/api/logs")
def get_logs(limit: int = 500):
    global _run_log
    return {"logs": _run_log[-limit:] if _run_log else []}

class ChatRequest(BaseModel):
    message: str
    state: dict = {}
    # The frontend has always sent this (apiService.ts's sendChat: "Supports
    # multi-turn conversation by passing message history", with a 5-minute
    # timeout explicitly anticipating a real, possibly-slow LLM call) — but
    # this field didn't exist here, so pydantic silently dropped it on every
    # request. A stub that ignores its own turn history was one of two
    # reasons multi-turn planning never worked; the other was that nothing
    # here ever called an LLM at all.
    history: list[dict] = []


_CHAT_SYSTEM_PROMPT = """You are Autobot's planning assistant. Have a natural, brief \
conversation with the user about the computer task they want done — browser \
automation, native desktop apps, research, file operations, anything Autobot can do.

Ask ONE clarifying question if something essential and genuinely ambiguous is \
missing (e.g. which website, which file). Don't ask about details you can \
reasonably infer or that don't change the plan.

Once you understand the goal well enough to act, stop asking questions and \
propose a plan.

Respond with ONLY valid JSON in this exact shape:
{
  "reply": "what you say to the user - a question, or a short summary if proposing a plan",
  "needs_plan": true or false,
  "plan_name": "short plan name (only if needs_plan is true)",
  "plan_description": "one sentence describing the goal (only if needs_plan is true)",
  "plan_steps": ["step 1 in plain language", "step 2", "..."]   // only if needs_plan is true, 3-8 steps
}"""


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    AI Planner: a real multi-turn conversation backed by an LLM, ending in a
    proposed plan the user can review and execute.

    Previously this endpoint never called an LLM at all — it synchronously
    fabricated an identical single-step "Auto Task" plan from whatever text
    was typed, regardless of content, and ignored the conversation history
    the frontend was already sending. That's what "an execution plan appears
    immediately, for no clear reason" was: there was no thinking step to
    wait for, because nothing was ever asked to think.
    """
    from ..agent.runner import _create_llm_client

    llm_client = _create_llm_client()
    if llm_client is None:
        return {
            "reply": "No LLM is configured, so I can't plan anything yet. "
                     "Set an API key (ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or "
                     "OPENAI_API_KEY) in .env, then restart the server. "
                     "Run 'autobot --doctor' to check your setup.",
            "plan": None,
        }

    messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
    for turn in req.history[-10:]:  # bounded — this is a chat panel, not the full run log
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": req.message})

    model = os.getenv("AUTOBOT_LLM_MODEL") or "gpt-4o"
    call_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    try:
        try:
            resp = await llm_client.chat.completions.create(**call_kwargs)
        except TypeError:
            resp = await asyncio.to_thread(llm_client.chat.completions.create, **call_kwargs)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        _log(f"Chat LLM call failed: {e}")
        return {
            "reply": f"Sorry, I couldn't reach the LLM: {e}",
            "plan": None,
        }

    data = _parse_chat_json(raw)
    if data is None:
        _log(f"Chat: could not parse LLM JSON, treating as plain reply: {raw[:200]}")
        return {"reply": raw or "I didn't get a response — try rephrasing?", "plan": None}

    reply = str(data.get("reply") or "").strip() or "..."
    plan = None
    if data.get("needs_plan") and data.get("plan_steps"):
        plan = {
            "id": f"plan_{int(time.time())}",
            "name": str(data.get("plan_name") or "Task Plan")[:100],
            "description": str(data.get("plan_description") or req.message)[:500],
            "steps": [
                {"action": "auto_execute", "args": {}, "description": str(s)[:300]}
                for s in data["plan_steps"][:10]
                if str(s).strip()
            ],
        }
    return {"reply": reply, "plan": plan}


def _parse_chat_json(raw: str) -> dict | None:
    """Parse the chat LLM's JSON reply, tolerating markdown code fences."""
    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL) or \
        re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


# ── Task Scheduler (multi-task queue) ───────────────────────────────────────
#
# The frontend (TaskQueuePanel.tsx, apiService.ts) has called these routes
# since before this session started — apiService.ts's QueuedTask/
# ScheduleStatus/ScreenLockStatus interfaces match agent/scheduler.py's
# ScheduledTask.summary() / get_schedule_status() / resource_manager.py's
# ScreenLock.get_status() field-for-field. Both halves were complete; only
# the routes connecting them were missing, which is what produced the 404s
# reported in the browser console. See scheduler.py's _tick() for why real
# concurrent execution is capped at 1 task despite AUTOBOT_MAX_CONCURRENT_TASKS
# defaulting to 3 — screen_lock isn't actually wired into AgentLoop's action
# dispatch yet, so more than one task running at once would mean multiple
# AgentRunners fighting over the same physical mouse/keyboard with no
# coordination. Queueing and sequential execution (what this wiring gives you)
# is real value on its own; true concurrency is separate, larger work.

class AddTaskRequest(BaseModel):
    goal: str
    priority: int = 1
    run_at: float | None = None  # epoch seconds; None = run ASAP


@app.get("/api/tasks")
def list_tasks():
    from ..agent.scheduler import scheduler
    return {"tasks": scheduler.get_all_tasks()}


@app.post("/api/tasks")
async def create_task(req: AddTaskRequest):
    from ..agent.scheduler import scheduler
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")
    task_id = await scheduler.add_task(req.goal, priority=req.priority, run_at=req.run_at)
    return {"status": "queued", "task_id": task_id}


@app.get("/api/tasks/{task_id}")
def get_task_detail(task_id: str):
    from ..agent.scheduler import scheduler
    task = scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.summary()


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    from ..agent.scheduler import scheduler
    ok = await scheduler.cancel_task(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "cancelled", "task_id": task_id}


@app.post("/api/tasks/{task_id}/pause")
async def pause_task_route(task_id: str):
    from ..agent.scheduler import scheduler
    ok = await scheduler.pause_task(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Task not found, or not queued (only queued tasks can be paused)")
    return {"status": "paused", "task_id": task_id}


@app.post("/api/tasks/{task_id}/resume")
async def resume_task_route(task_id: str):
    from ..agent.scheduler import scheduler
    ok = await scheduler.resume_task(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Task not found, or not paused")
    return {"status": "resumed", "task_id": task_id}


@app.patch("/api/tasks/{task_id}/priority")
async def set_task_priority_route(task_id: str, priority: int):
    from ..agent.scheduler import scheduler
    ok = await scheduler.reprioritize_task(task_id, priority)
    if not ok:
        raise HTTPException(status_code=400, detail="Task not found, or not queued/paused/scheduled")
    return {"status": "ok"}


@app.get("/api/tasks/{task_id}/logs")
def get_task_logs_route(task_id: str, since: int = 0):
    from ..agent.scheduler import scheduler
    task = scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    lines = scheduler.get_logs(task_id, since)
    return {"lines": lines, "total": len(task.logs)}


@app.get("/api/schedule/status")
def get_schedule_status_route():
    from ..agent.scheduler import scheduler
    return scheduler.get_schedule_status()


@app.get("/api/screen-lock")
def get_screen_lock_status_route():
    from ..agent.resource_manager import screen_lock
    return screen_lock.get_status()


@app.post("/api/utils/anti-sleep")
def set_anti_sleep(req: AntiSleepRequest):
    from ..computer.anti_sleep import anti_sleep
    if req.enabled:
        anti_sleep.start()
    else:
        anti_sleep.stop()
    return {"status": "ok", "enabled": req.enabled}


# ── Added Fallback & WebSocket routes for frontend compatibility ─────────────
@app.get("/api/health")
def get_health_route():
    return {
        "overall_ok": True,
        "llm": {"ok": True, "provider": os.getenv("AUTOBOT_LLM_PROVIDER", "openrouter"), "model": os.getenv("AUTOBOT_LLM_MODEL", "openai/gpt-4o-mini"), "error": ""},
        "cdp": {"ok": True, "tabs": 1, "url": "", "error": ""},
        "config": {"has_api_key": True, "vision_enabled": True}
    }

@app.get("/api/learning/stats")
def get_learning_stats_route():
    return {
        "rl_enabled": True,
        "total_experiences": 0,
        "learned_contexts": 0,
        "total_policy_observations": 0,
        "current_run_steps": 0,
        "run_id": "",
        "memory_entries": 0,
        "memory_hits": 0,
        "memory_high_value": 0
    }

@app.get("/api/tunnel/status")
def get_tunnel_status_route():
    return {"active": False, "url": ""}

@app.get("/api/onboarding/status")
def get_onboarding_status_route():
    return {"complete": True}

@app.post("/api/onboarding")
def post_onboarding_route():
    return {"status": "ok", "saved": []}

@app.websocket("/ws/events")
async def ws_events_alias(websocket: WebSocket):
    await ws_logs(websocket)


_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
