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
from typing import Any

from autobot.integrations.cli_exec import classify_error, run_cli, strip_flag, unknown_flag

_DEFAULT_TIMEOUT = 600.0   # headless coding turns can run long

WRITE_CAPABLE_MODES = {"acceptEdits", "bypassPermissions", "auto", "dontAsk"}

# Older Claude Code builds don't know this flag; if one rejects it we retry without.
_PERMISSION_PROMPTS_FLAG = ["--permission-prompts", "none"]
_VALUE_FLAGS = {"--permission-prompts", "--allowedTools", "--disallowedTools", "--tools", "--json-schema",
                "--model", "--append-system-prompt", "--add-dir", "--resume", "--output-format", "--permission-mode"}
# Never silently dropped: without these the call would run with MORE power than asked.
_SAFETY_FLAGS = {"--permission-mode", "--disallowedTools", "--output-format"}


def is_available() -> bool:
    """True if the `claude` CLI is installed and on PATH (claude.exe or npm's claude.cmd)."""
    return shutil.which("claude") is not None


def _build_args(
    permission_mode: str,
    allowed_tools: list[str] | None,
    continue_session: bool,
    resume_session_id: str | None,
    disallowed_tools: list[str] | None,
    tools: list[str] | None,
    json_schema: dict | str | None,
    model: str | None,
    append_system_prompt: str | None,
    no_session_persistence: bool,
    add_dirs: list[str] | None,
    with_permission_prompts_flag: bool,
) -> list[str]:
    args = ["claude", "-p", "--output-format", "json", "--permission-mode", permission_mode]
    if with_permission_prompts_flag:
        args += _PERMISSION_PROMPTS_FLAG
    if allowed_tools:
        args += ["--allowedTools", ",".join(allowed_tools)]
    if disallowed_tools:
        args += ["--disallowedTools", ",".join(disallowed_tools)]
    if tools is not None:
        args += ["--tools", ",".join(tools) if tools else ""]
    if json_schema is not None:
        args += ["--json-schema", json_schema if isinstance(json_schema, str) else json.dumps(json_schema)]
    if model:
        args += ["--model", model]
    if append_system_prompt:
        args += ["--append-system-prompt", append_system_prompt]
    if no_session_persistence:
        args += ["--no-session-persistence"]
    for d in add_dirs or []:
        args += ["--add-dir", d]
    if resume_session_id:
        args += ["--resume", resume_session_id]
    elif continue_session:
        args += ["--continue"]
    return args


def run_headless(
    prompt: str,
    cwd: str | None = None,
    permission_mode: str = "plan",
    allowed_tools: list[str] | None = None,
    continue_session: bool = False,
    resume_session_id: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    *,
    disallowed_tools: list[str] | None = None,
    tools: list[str] | None = None,
    json_schema: dict | str | None = None,
    model: str | None = None,
    append_system_prompt: str | None = None,
    no_session_persistence: bool = False,
    add_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run one headless Claude Code turn and return
    {"ok", "data", "error", "error_class"}. Never raises.

    permission_mode: "plan" (read-only, default) | "acceptEdits" |
    "bypassPermissions" | whatever else the installed version accepts.
    disallowed_tools: hard denials that hold even in write-capable modes,
    e.g. ["Bash(git push*)", "Bash(kaggle competitions submit*)"].
    tools=[] disables all tools (pure text/JSON answer — used when Claude is
    the butler's decision-maker rather than a coding worker).
    json_schema: structured output; the parsed object comes back in
    data["structured_output"] (verified against Claude Code 2.1).

    The prompt is sent on STDIN, not as an argument: on Windows an
    npm-installed `claude.cmd` runs through cmd.exe, which cuts arguments
    at the first newline and has a ~8K command-line limit.
    """
    if not prompt or not prompt.strip():
        return {"ok": False, "data": None, "error": "run_headless: prompt is required", "error_class": "other"}
    if not is_available():
        return {
            "ok": False, "data": None, "error_class": "not_installed",
            "error": "claude CLI not found on PATH. Install: npm install -g @anthropic-ai/claude-code",
        }
    if cwd and not os.path.isdir(cwd):
        return {"ok": False, "data": None, "error_class": "other",
                "error": f"cwd does not exist or is not a directory: {cwd}"}

    common = dict(
        permission_mode=permission_mode, allowed_tools=allowed_tools, continue_session=continue_session,
        resume_session_id=resume_session_id, disallowed_tools=disallowed_tools, tools=tools,
        json_schema=json_schema, model=model, append_system_prompt=append_system_prompt,
        no_session_persistence=no_session_persistence, add_dirs=add_dirs,
    )
    argv = _build_args(with_permission_prompts_flag=True, **common)
    res = run_cli(argv, input_text=prompt, cwd=cwd, timeout=timeout)
    # An older Claude Code that doesn't know a newer flag (--permission-prompts,
    # --json-schema, --tools, --no-session-persistence...) rejects the whole call.
    # Drop the flag it names and retry, rather than failing every task.
    for _ in range(4):
        if res.ok or res.timed_out or res.not_found:
            break
        flag = unknown_flag(res.stderr + " " + res.stdout)
        if not flag or flag not in argv or flag in _SAFETY_FLAGS:
            break
        argv = strip_flag(argv, flag, _VALUE_FLAGS)
        res = run_cli(argv, input_text=prompt, cwd=cwd, timeout=timeout)

    if res.not_found:
        return {"ok": False, "data": None, "error": res.stderr, "error_class": "not_installed"}
    if res.timed_out:
        return {"ok": False, "data": res.stdout or None, "error": f"claude -p timed out after {timeout}s",
                "error_class": "timeout"}

    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    parsed: Any = None
    if out:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None

    if res.returncode != 0:
        message = err or (parsed.get("result") if isinstance(parsed, dict) else None) or out \
            or f"claude exited with code {res.returncode}"
        return {"ok": False, "data": parsed if parsed is not None else (out or None),
                "error": str(message), "error_class": classify_error(f"{err} {out}")}

    if parsed is None:
        return {"ok": True, "data": {"result": out}, "error": "warning: could not parse JSON output",
                "error_class": ""}
    if isinstance(parsed, dict) and parsed.get("is_error"):
        message = str(parsed.get("result") or parsed.get("subtype") or "claude reported an error")
        return {"ok": False, "data": parsed, "error": message,
                "error_class": classify_error(f"{message} {parsed.get('api_error_status') or ''}")}
    return {"ok": True, "data": parsed, "error": "", "error_class": ""}
