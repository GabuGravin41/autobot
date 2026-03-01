"""
autobot/web/app.py — FastAPI application bridging the React frontend
to the core AutomationEngine / LLMBrain backend.

API surface (all under /api):
  GET  /api/status            → engine health, run state, browser mode
  GET  /api/adapters          → list all registered adapters + their actions
  GET  /api/runs              → run history list (from runs/ folder)
  GET  /api/run/{run_id}      → details + logs for a specific run
  POST /api/run/{run_id}/cancel  → cancel an active run
  POST /api/plan/text         → build a WorkflowPlan from natural language
  POST /api/plan/run          → execute a WorkflowPlan (async)
  GET  /api/settings          → read current settings
  POST /api/settings          → update .env-based settings
  POST /api/chat              → chat with the LLM planner, get a plan back

WS  /ws/logs                  → real-time log streaming for the active run
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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ── Core autobot imports ──────────────────────────────────────────────────────
from ..engine import AutomationEngine, TaskStep, WorkflowPlan
from ..planner import build_plan_from_text
from ..llm_brain import LLMBrain
from ..workflows import (
    builtin_workflows,
    code_iteration_workflow,
    console_fix_assist_workflow,
    kaggle_submit_workflow,
    leetcode_solve_workflow,
    open_whatsapp_stay_workflow,
    portfolio_workflow,
    research_paper_workflow,
    tool_call_stress_workflow,
    website_builder_workflow,
)

# ── Module-level shared state ─────────────────────────────────────────────────

_engine: AutomationEngine | None = None
_brain: LLMBrain | None = None
_run_log: list[str] = []
_active_run_id: str | None = None
_run_status: str = "idle"   # idle | running | done | failed | cancelled
_ws_clients: set[WebSocket] = set()
_event_loop: asyncio.AbstractEventLoop | None = None
# Human input: when request_human_input runs, callback sets this and waits; POST /api/human_input clears it
_human_input_pending: dict[str, Any] | None = None

# All actions the AI planner is allowed to use
ALLOWED_ACTIONS = [
    "open_url", "search_google", "browser_fill", "browser_click",
    "browser_press", "browser_scroll", "browser_snapshot",
    "browser_get_status", "browser_get_url", "browser_set_mode",
    "adapter_call", "adapter_list_actions", "adapter_set_policy",
    "adapter_prepare_sensitive", "adapter_confirm_sensitive",
    "clipboard_set", "clipboard_get", "screenshot", "wait",
    "desktop_type", "desktop_hotkey", "desktop_click", "desktop_move",
    "desktop_switch_window", "run_command", "open_app", "open_path",
    "request_human_help", "request_human_input", "log",
]


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

def _human_input_callback(prompt: str, state_key: str) -> str:
    """Called from engine thread when a step needs password/token. Blocks until frontend submits value."""
    global _human_input_pending
    result_holder: list[str] = [""]
    evt = threading.Event()
    _human_input_pending = {"prompt": prompt, "key": state_key, "event": evt, "result": result_holder}
    _log(f"!!! Human input required: {state_key} — check the React UI to enter value !!!")
    if _event_loop and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(json.dumps({"type": "human_input_required", "prompt": prompt, "key": state_key})), _event_loop)
    ok = evt.wait(timeout=300)
    _human_input_pending = None
    if not ok:
        _log("Human input timed out after 300s; using empty string.")
    return (result_holder[0] or "").strip()


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _engine, _brain, _event_loop
    _event_loop = asyncio.get_event_loop()
    _log("Autobot engine starting...")
    _engine = AutomationEngine(logger=_log, human_input_callback=_human_input_callback)
    _brain = LLMBrain(logger=_log)
    if _brain.enabled:
        _log(f"LLM Brain enabled: provider={_brain.provider}, model={_brain.model_name}")
    else:
        _log("LLM Brain disabled (no API key configured). Using text-based planner fallback.")
    yield
    _log("Autobot engine shutting down.")
    if _engine:
        _engine.close()


app = FastAPI(title="Autobot API", version="1.0.0", lifespan=lifespan)

# Allow Vite dev server (various ports) + any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request models ───────────────────────────────────────────────────

class TextPlanRequest(BaseModel):
    task: str

class RunPlanRequest(BaseModel):
    plan: dict[str, Any]

class ChatRequest(BaseModel):
    message: str
    state: dict[str, Any] = {}

class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    browser_mode: str | None = None


class RunWorkflowRequest(BaseModel):
    workflow_id: str
    topic: str = ""


class HumanInputSubmit(BaseModel):
    key: str
    value: str


# ── Plan conversion helpers ───────────────────────────────────────────────────

def _dict_to_plan(data: dict[str, Any]) -> WorkflowPlan:
    steps = [
        TaskStep(
            action=str(s.get("action", "")),
            args=dict(s.get("args") or {}),
            description=str(s.get("description", "")),
            save_as=s.get("save_as") or None,
            retries=int(s.get("retries") or 0),
            continue_on_error=bool(s.get("continue_on_error", False)),
            target_node=s.get("target_node") or None,
        )
        for s in (data.get("steps") or [])
        if isinstance(s, dict) and s.get("action")
    ]
    return WorkflowPlan(
        name=str(data.get("name") or "unnamed"),
        description=str(data.get("description") or ""),
        steps=steps,
    )


def _plan_to_dict(plan: WorkflowPlan, plan_id: str | None = None) -> dict[str, Any]:
    return {
        "id": plan_id or f"plan_{int(time.time())}",
        "name": plan.name,
        "description": plan.description,
        "steps": [
            {
                "action": s.action,
                "args": s.args,
                "description": s.description,
                "save_as": s.save_as,
                "retries": s.retries,
                "continue_on_error": s.continue_on_error,
                "target_node": s.target_node,
            }
            for s in plan.steps
        ],
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Engine health, browser state, and current run status."""
    browser_active = False
    browser_mode = "unknown"
    current_url = "none"
    if _engine:
        try:
            browser_active = _engine.browser.is_active()
            browser_mode = _engine.browser.mode
            if browser_active and browser_mode != "human_profile":
                current_url = _engine.browser.get_url()
        except Exception:
            pass

    human_input = None
    if _human_input_pending:
        human_input = {"prompt": _human_input_pending.get("prompt", ""), "key": _human_input_pending.get("key", "")}
    return {
        "status": "ok",
        "run_status": _run_status,
        "active_run_id": _active_run_id,
        "browser": {
            "active": browser_active,
            "mode": browser_mode,
            "url": current_url,
        },
        "llm_enabled": bool(_brain and _brain.enabled),
        "llm_provider": os.getenv("AUTOBOT_LLM_PROVIDER", "none"),
        "llm_model": os.getenv("AUTOBOT_LLM_MODEL", "default"),
        "log_lines": len(_run_log),
        "human_input_pending": human_input,
    }


@app.get("/api/adapters")
def get_adapters():
    """List all registered adapters and their available actions."""
    if not _engine:
        return {"adapters": []}
    lib = _engine.get_adapter_library()
    result = []
    for name, info in lib.items():
        actions = list(info.get("actions", {}).keys())
        result.append({
            "name": name,
            "description": info.get("description", ""),
            "actions": actions,
            "telemetry": info.get("telemetry", {}),
        })
    return {"adapters": result}


@app.get("/api/runs")
def get_runs():
    """List all completed/active run folders from the runs/ directory."""
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
                    # Normalize field names for React frontend
                    data["planName"] = data.get("plan_name", "unnamed")
                    data["timestamp"] = data.get("started_at", "unknown")
                    data["status"] = "success" if data.get("success") else "failed"
                    data["stepsCompleted"] = data.get("completed_steps", 0)
                    data["totalSteps"] = data.get("total_steps", 0)
                    
                    # Compute progress
                    if data["totalSteps"] > 0:
                        data["progress"] = int((data["stepsCompleted"] / data["totalSteps"]) * 100)
                    else:
                        data["progress"] = 100
                    
                    # Add a snippet of the latest logs for the UI
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


@app.get("/api/run/{run_id}")
def get_run(run_id: str):
    """Get details and logs for a specific run."""
    # Active run: return live data
    if run_id == _active_run_id:
        return {
            "id": run_id,
            "planName": "Active Run",
            "status": _run_status,
            "stepsCompleted": len([l for l in _run_log if "Executing:" in l]),
            "totalSteps": 0,
            "logs": list(_run_log),
            "active": True,
        }
    # Historical run
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
    elif "steps" in data:
        # Fallback to step logs if console.log is missing
        data["logs"] = [f"Step {i+1}: {s.get('description', '')}" for i, s in enumerate(data.get("steps", []))]
    
    # Normalize for frontend
    data["planName"] = data.get("plan_name", "historical_run")
    data["timestamp"] = data.get("started_at", "unknown")
    data["status"] = "success" if data.get("success") else "failed"
    data["stepsCompleted"] = data.get("completed_steps", 0)
    data["totalSteps"] = data.get("total_steps", 0)
    
    return data


@app.delete("/api/run/{run_id}")
def delete_run(run_id: str):
    """Delete a run folder and its history."""
    runs_root = Path(__file__).resolve().parent.parent.parent / "runs"
    run_dir = runs_root / run_id
    if run_dir.exists() and run_dir.is_dir():
        import shutil
        shutil.rmtree(run_dir)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Run not found")


@app.get("/api/browser/screenshot")
def get_browser_screenshot():
    """Capture a live screenshot: browser tab or full desktop (so user can see current state of the screen)."""
    if not _engine:
        raise HTTPException(status_code=400, detail="Engine not initialized")

    tmp_path = Path.cwd() / "tmp" / "live_screenshot.png"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _engine.browser.screenshot(str(tmp_path))
        if tmp_path.exists():
            return FileResponse(tmp_path, media_type="image/png")
    except Exception as e:
        _log(f"Screenshot error: {e}")
        try:
            import pyautogui
            pyautogui.screenshot(imageFilename=str(tmp_path))
            if tmp_path.exists():
                return FileResponse(tmp_path, media_type="image/png")
        except Exception as e2:
            _log(f"Desktop screenshot fallback error: {e2}")
        raise HTTPException(status_code=500, detail=f"Failed to capture screenshot: {e}")

    raise HTTPException(status_code=503, detail="Browser not ready for screenshot")


@app.post("/api/plan/text")
def plan_from_text(req: TextPlanRequest):
    """Convert a natural-language task string into a structured WorkflowPlan."""
    try:
        plan = build_plan_from_text(req.task)
        return {"plan": _plan_to_dict(plan)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/plan/run")
def run_plan_endpoint(req: RunPlanRequest):
    """
    Execute a WorkflowPlan asynchronously.
    Returns immediately with a run_id; poll /api/status or /api/run/{run_id} for progress.
    """
    global _active_run_id, _run_status, _run_log

    if _run_status == "running":
        raise HTTPException(status_code=409, detail="A plan is already running. Cancel it first.")

    engine = _engine
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready.")

    try:
        plan = _dict_to_plan(req.plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {e}")

    if not plan.steps:
        raise HTTPException(status_code=400, detail="Plan has no steps to execute.")

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    _active_run_id = run_id
    _run_status = "running"
    _run_log.clear()
    _log(f"▶ Starting plan: '{plan.name}' ({len(plan.steps)} steps)")

    def _run_in_thread():
        global _run_status
        try:
            engine.run_plan(plan)
            _run_status = "done"
            _log(f"✓ Plan '{plan.name}' completed successfully.")
        except Exception as ex:
            _run_status = "failed"
            _log(f"✗ Plan failed: {ex}")

    threading.Thread(target=_run_in_thread, daemon=True, name=f"plan-{run_id}").start()
    return {"run_id": run_id, "status": "started", "plan_name": plan.name}


# ── Builtin workflows (same as React frontend / former Tkinter UI) ─────────────

WORKFLOW_LIST: list[dict[str, Any]] = [
    {"id": "open_whatsapp_stay", "name": "Open WhatsApp Stay", "description": "Open WhatsApp Web and leave it open.", "topic_label": "Phone (optional)"},
    {"id": "website_builder", "name": "Website Builder", "description": "Open VS Code, CASA, Grok, and search for layout ideas.", "topic_label": "Topic"},
    {"id": "research_paper", "name": "Research Paper", "description": "Search, ask Grok for LaTeX, open Overleaf, compile.", "topic_label": "Topic"},
    {"id": "portfolio", "name": "Portfolio", "description": "Google AI Studio sign-in, Gemini portfolio, save, git, Vercel.", "topic_label": "Topic (optional)"},
    {"id": "leetcode_solve", "name": "LeetCode Solve", "description": "Solve one LeetCode problem with Grok (slug).", "topic_label": "Problem slug (e.g. two-sum)"},
    {"id": "kaggle_submit", "name": "Kaggle Submit", "description": "Open competition, run notebook, submit, wait for score.", "topic_label": "Competition slug (e.g. titanic)"},
    {"id": "console_fix_assist", "name": "Console Fix Assist", "description": "Capture console errors and open Grok with context.", "topic_label": "URL (optional)"},
    {"id": "code_iteration", "name": "Code Iteration", "description": "Run tests, open Grok with failure context.", "topic_label": "Test command (e.g. pytest)"},
    {"id": "tool_call_stress", "name": "Tool Call Stress", "description": "WhatsApp → Google Docs → Grok → Overleaf chain.", "topic_label": "phone|docs_url|path|message"},
]


@app.get("/api/workflows")
def list_workflows():
    """Return builtin workflow definitions for the React frontend."""
    return {"workflows": WORKFLOW_LIST}


def _workflow_plan(workflow_id: str, topic: str) -> WorkflowPlan:
    """Build a WorkflowPlan from workflow_id and topic (same logic as Tkinter UI)."""
    key = (workflow_id or "").strip()
    t = (topic or "").strip()
    if key == "open_whatsapp_stay":
        return open_whatsapp_stay_workflow(phone=t)
    if key == "website_builder":
        return website_builder_workflow(t)
    if key == "research_paper":
        return research_paper_workflow(t)
    if key == "portfolio":
        return portfolio_workflow(t)
    if key == "leetcode_solve":
        return leetcode_solve_workflow(t or "two-sum")
    if key == "kaggle_submit":
        return kaggle_submit_workflow(t or "titanic", 1200)
    if key == "console_fix_assist":
        return console_fix_assist_workflow(t or "http://localhost:3000")
    if key == "code_iteration":
        return code_iteration_workflow(t or "pytest")
    if key == "tool_call_stress":
        parts = [p.strip() for p in t.split("|")] if t else []
        while len(parts) < 4:
            parts.append("")
        return tool_call_stress_workflow(whatsapp_phone=parts[0], docs_existing_url=parts[1], download_check_path=parts[2], outgoing_message=parts[3] or "Autobot test")
    raise ValueError(f"Unknown workflow: {workflow_id}")


@app.post("/api/workflows/run")
def run_workflow(req: RunWorkflowRequest):
    """Run a builtin workflow by id with optional topic. Returns run_id."""
    try:
        plan = _workflow_plan(req.workflow_id, req.topic)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return run_plan_endpoint(RunPlanRequest(plan=_plan_to_dict(plan)))


@app.get("/api/human_input")
def get_pending_human_input():
    """Return current pending human input request, if any (for polling or initial load)."""
    global _human_input_pending
    if not _human_input_pending:
        return {"pending": False}
    return {
        "pending": True,
        "prompt": _human_input_pending.get("prompt", ""),
        "key": _human_input_pending.get("key", ""),
    }


@app.post("/api/human_input")
def submit_human_input(req: HumanInputSubmit):
    """Submit a value for a pending request_human_input step. Unblocks the engine."""
    global _human_input_pending
    if not _human_input_pending:
        raise HTTPException(status_code=400, detail="No human input request pending.")
    if _human_input_pending.get("key") != req.key:
        raise HTTPException(status_code=400, detail=f"Key mismatch. Expected: {_human_input_pending.get('key')}")
    result = _human_input_pending.get("result")
    evt = _human_input_pending.get("event")
    if isinstance(result, list):
        result[:] = [req.value]
    if evt is not None:
        evt.set()
    return {"status": "submitted", "key": req.key}


@app.post("/api/run/{run_id}/cancel")
def cancel_run(run_id: str):
    """Cancel the currently active run."""
    global _run_status
    if run_id != _active_run_id:
        raise HTTPException(status_code=400, detail="Run ID does not match active run.")
    if _run_status != "running":
        raise HTTPException(status_code=400, detail=f"Run is not active (status: {_run_status}).")
    if _engine:
        _engine.cancel()
    _run_status = "cancelled"
    _log("⚠ Run cancelled by user.")
    return {"status": "cancelled", "run_id": run_id}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Send a message to the AI planner.
    
    Flow:
    1. If LLM is configured → use LLMBrain.generate_plan_draft() to get an AI plan
    2. Fallback → use text-based planner (build_plan_from_text)
    
    Returns: { reply: str, plan: PlanDict | null }
    """
    brain = _brain

    if brain and brain.enabled:
        try:
            # Build tool catalog from registered adapters so LLM knows available tools
            tool_catalog = "## GENERAL BROWSER TOOLS (Any website)\n"
            tool_catalog += "### Mode: devtools (Fast, DOM Access)\n"
            tool_catalog += "- browser_get_content(): COPY/CAPTURE all text from page. Use for extraction.\n"
            tool_catalog += "- browser_click_text(text): Click button/link by its name. Very robust.\n"
            tool_catalog += "- browser_click(selector): Click element by CSS selector.\n"
            tool_catalog += "- browser_fill(selector, text): Type into input by CSS selector.\n"
            tool_catalog += "- browser_snapshot(): Get visible text-tree of buttons/inputs.\n"
            tool_catalog += "### Mode: human_profile (Stealth, Simulation)\n"
            tool_catalog += "- desktop_type(text): Simulate typing via system keyboard. Bypasses detection.\n"
            tool_catalog += "- desktop_press(key): Simulate key press (e.g. 'enter').\n"
            tool_catalog += "### Global Tools (Any Mode)\n"
            tool_catalog += "- open_url(url): Navigate to site.\n"
            tool_catalog += "- browser_set_mode(mode): Switch capability between 'devtools' and 'human_profile'.\n"
            tool_catalog += "- browser_press(key): Press key 'enter' on browser window.\n"
            tool_catalog += "- search_google(query): Search and open top result.\n\n"
            
            tool_catalog += "## SERVICE ADAPTERS (Use ONLY for these specific services)\n"
            if _engine:
                lib = _engine.get_adapter_library()
                catalog_lines: list[str] = []
                for adapter_name, info in lib.items():
                    adapter_desc = info.get("description", "")
                    catalog_lines.append(f"### Adapter: {adapter_name}")
                    if adapter_desc:
                        catalog_lines.append(f"Description: {adapter_desc}")
                    catalog_lines.append("Actions:")
                    for action_name, action_info in info.get("actions", {}).items():
                        desc = action_info.get("description", "No description")
                        catalog_lines.append(f"  - {action_name}: {desc}")
                    catalog_lines.append("")
                if catalog_lines:
                    tool_catalog += "\n".join(catalog_lines)

            draft = brain.generate_plan_draft(
                user_prompt=req.message,
                allowed_actions=ALLOWED_ACTIONS,
                max_steps=10,
                tool_catalog=tool_catalog or None,
            )

            if draft and draft.steps:
                # Convert PlanDraft → plan dict (PlanDraft has .title, .summary, .steps)
                plan_dict: dict[str, Any] = {
                    "id": f"plan_{int(time.time())}",
                    "name": draft.title or "AI Plan",
                    "description": draft.summary or req.message,
                    "steps": [
                        {
                            "action": s.action,
                            "args": s.args,
                            "description": s.description,
                            "save_as": s.save_as,
                            "retries": s.retries,
                            "continue_on_error": s.continue_on_error,
                            "target_node": s.target_node,
                        }
                        for s in draft.steps
                    ],
                }
                reply = (
                    f"I've created a plan: **{plan_dict['name']}**\n\n"
                    f"{draft.summary}\n\n"
                    f"Ready to execute {len(draft.steps)} step(s)?"
                )
                return {"reply": reply, "plan": plan_dict}
            else:
                return {
                    "reply": "I couldn't generate a plan for that request. Try rephrasing, or be more specific.",
                    "plan": None,
                }
        except Exception as e:
            _log(f"LLM chat error: {e}")
            # Fall through to text-based planner

    # Fallback: rule-based text planner
    try:
        plan = build_plan_from_text(req.message)
        plan_dict = _plan_to_dict(plan)
        if plan.steps:
            reply = (
                f"Here's a workflow for: **{plan.name}**\n\n"
                f"{plan.description}\n\n"
                f"This plan has {len(plan.steps)} step(s). Ready to execute?"
            )
            return {"reply": reply, "plan": plan_dict}
        else:
            return {
                "reply": (
                    "I understand what you're asking, but I need more specifics. "
                    "Try commands like: \"search for Python tutorials\", "
                    "\"open WhatsApp\", or \"run stress test\". "
                    "Configure an LLM API key in Settings for full AI planning."
                ),
                "plan": None,
            }
    except Exception:
        return {
            "reply": (
                f'I heard: "{req.message}". '
                "To get a proper plan, configure an LLM API key in Settings, "
                "or try a command like \"search for X\" or \"open Y\"."
            ),
            "plan": None,
        }


@app.get("/api/settings")
def get_settings():
    """Return current configuration (no secret values exposed)."""
    return {
        "llm_provider": os.getenv("AUTOBOT_LLM_PROVIDER", "none"),
        "llm_model": os.getenv("AUTOBOT_LLM_MODEL", ""),
        "browser_mode": os.getenv("AUTOBOT_BROWSER_MODE", "human_profile"),
        "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_gemini_key": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "llm_enabled": bool(_brain and _brain.enabled),
    }


@app.post("/api/settings")
def update_settings(req: SettingsUpdate):
    """Update .env settings at runtime."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    updates: dict[str, str] = {}
    if req.llm_provider is not None:
        updates["AUTOBOT_LLM_PROVIDER"] = req.llm_provider
    if req.llm_model is not None:
        updates["AUTOBOT_LLM_MODEL"] = req.llm_model
    if req.browser_mode is not None:
        updates["AUTOBOT_BROWSER_MODE"] = req.browser_mode
    if req.openrouter_api_key:
        updates["OPENROUTER_API_KEY"] = req.openrouter_api_key
        # Automatically set provider if user is saving an OpenRouter key
        if not req.llm_provider:
            updates["AUTOBOT_LLM_PROVIDER"] = "openrouter"
    if req.openai_api_key:
        updates["OPENAI_API_KEY"] = req.openai_api_key

    if updates:
        # Update environment variables immediately for the running process
        for key, val in updates.items():
            os.environ[key] = val

        # Also persist to .env file
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

        # Re-initialise LLM brain with new settings
        global _brain
        _brain = LLMBrain(logger=_log)
        if _brain.enabled:
            _log(f"LLM Brain re-initialised: provider={_brain.provider}, model={_brain.model_name}")
        else:
            _log("LLM Brain re-initialised but not enabled (check API key).")

    return {"status": "updated", "keys_changed": list(updates.keys())}


# ── Logs (polling fallback when WebSocket is unavailable) ─────────────────────

@app.get("/api/logs")
def get_logs(limit: int = 500):
    """Return recent log lines for polling fallback. Use when WebSocket is unreliable."""
    n = max(1, min(2000, limit))
    return {"logs": list(_run_log[-n:])}


# ── WebSocket: real-time log streaming ───────────────────────────────────────

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    # Send the current log backlog immediately on connect
    for line in list(_run_log):
        try:
            await websocket.send_text(line)
        except Exception:
            break
    try:
        while True:
            await asyncio.sleep(30)
            # Ping each client; remove dead ones so one bad client doesn't break others
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


# ── Serve built React frontend ────────────────────────────────────────────────

_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
