"""
Claude Code Bridge — subprocess wrapper around Claude Code's official
headless/non-interactive mode (`claude -p`).

Why headless instead of UI automation: Claude Code ships a documented
scripting mode built exactly for this ("claude -p PROMPT --output-format
json"). Autobot doesn't need to open a chat window, paste text into it,
and read pixels or DOM back out — it runs Claude Code as a subprocess and
gets structured JSON back directly. Same "API before UI automation"
principle as kaggle_bridge.py.

Every public function returns the same shape as
autobot/browser/extension_bridge.py's ExtensionBridge.run():
    {"ok": bool, "data": <dict | str | None>, "error": str}
and never raises.

No shell=True anywhere in this file: the prompt, cwd, and every other
caller-supplied value go straight into an argv list passed to subprocess,
never through a shell — so nothing an LLM puts in `prompt` can be
interpreted as shell syntax.

Safety, deliberately explicit: `permission_mode` defaults to "plan" — read
only, Claude Code will not write files or run tools regardless of what it
decides to do. Callers (CoreLoop) must opt in to "acceptEdits" or
"bypassPermissions" to allow real writes, and CoreLoop's own risk
classifier (_classify_risk() in core_loop.py) treats that opt-in as
DANGER-tier, gated the same way shell execution is. This module does not
make that safety decision itself; it forwards whatever permission_mode the
caller asked for and reports what actually happened.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

_DEFAULT_TIMEOUT = 600.0   # headless coding turns can run long

# permission_mode values that let Claude Code write files / run tools
# without prompting. Kept here (not duplicated as a string literal) so
# core_loop.py's risk classifier and this module can never drift apart.
WRITE_CAPABLE_MODES = {"acceptEdits", "bypassPermissions"}


def is_available() -> bool:
    """True if the `claude` CLI is installed and on PATH."""
    return shutil.which("claude") is not None


def run_headless(
    prompt: str,
    cwd: str | None = None,
    permission_mode: str = "plan",
    allowed_tools: list[str] | None = None,
    continue_session: bool = False,
    resume_session_id: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Run one headless Claude Code turn and return its result.

    permission_mode:
      "plan"             — read-only; Claude Code will not write files or
                            run tools without asking, and headless mode
                            can't answer a prompt, so it just won't do it.
                            The safe default.
      "acceptEdits"       — writes files without prompting.
      "bypassPermissions" — writes files AND runs shell/tool calls without
                             prompting. Most capable, least safe.
      "default"           — Claude Code's normal interactive gating, which
                             will hang forever in headless mode. Avoid.

    resume_session_id / continue_session let a caller pick up a prior
    headless run's conversation (e.g. "read the Kaggle code" as turn one,
    "now write the improved version" as turn two) instead of re-explaining
    context from scratch every call.
    """
    if not prompt or not prompt.strip():
        return {"ok": False, "data": None, "error": "run_headless: prompt is required"}
    if not is_available():
        return {
            "ok": False,
            "data": None,
            "error": "claude CLI not found on PATH. Install: npm install -g @anthropic-ai/claude-code",
        }
    if cwd and not os.path.isdir(cwd):
        # Ported from antigravity_bridge.py (Round 6 adversarial-review fix,
        # flagged there as "identical bug, out of scope for that round's
        # diff" — ported here Round 8 since orchestrator_dispatch.py now
        # calls both bridges from tracked projects' working_dir, making a
        # stale/deleted directory a real, not just theoretical, path).
        # subprocess.run(cwd=...) raises FileNotFoundError for a missing
        # directory, which is the SAME exception type the `except
        # FileNotFoundError` clause below catches for a completely
        # different reason (the `claude` binary itself vanishing from PATH
        # between the is_available() check and exec) — without this
        # earlier check, a bad cwd would be caught by that clause and
        # misreported as "claude CLI not found on PATH", actively
        # misleading a caller who has claude installed just fine.
        return {
            "ok": False,
            "data": None,
            "error": f"cwd does not exist or is not a directory: {cwd}",
        }

    args = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--permission-mode", permission_mode,
        "--permission-prompts", "none",
    ]
    if allowed_tools:
        args += ["--allowedTools", ",".join(allowed_tools)]
    if resume_session_id:
        args += ["--resume", resume_session_id]
    elif continue_session:
        args += ["--continue"]

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "data": None, "error": f"claude -p timed out after {timeout}s"}
    except FileNotFoundError:
        return {"ok": False, "data": None, "error": "claude CLI not found on PATH."}
    except Exception as e:
        return {"ok": False, "data": None, "error": f"{type(e).__name__}: {e}"}

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if proc.returncode != 0:
        return {"ok": False, "data": out or None, "error": err or f"claude exited with code {proc.returncode}"}

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        # --output-format json should always be valid JSON on a clean exit,
        # but don't crash the caller if a future CLI version ever changes
        # that — surface the raw text instead of failing the whole call.
        return {"ok": True, "data": {"result": out}, "error": "warning: could not parse JSON output"}

    return {"ok": True, "data": parsed, "error": ""}
