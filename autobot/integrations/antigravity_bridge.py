"""
Antigravity Bridge — subprocess wrapper around Google Antigravity's official
headless CLI mode (`agy -p`).

Mirrors autobot/integrations/claude_code_bridge.py deliberately closely:
same "API before UI automation" reasoning applies here even more strongly,
because Antigravity's chat panel (like VS Code's) turned out to be
reachable through this session's computer-use tools only at 'full' tier
for Antigravity itself — but the underlying design principle this project
follows (see DESIGN_PHILOSOPHY.md) is to prefer a real CLI/API over GUI
automation whenever one exists, not just when GUI automation happens to be
blocked. Antigravity ships one: confirmed against the official docs
(antigravity.google/docs/cli/headless/, checked Sep 2026) — `agy -p PROMPT
--output-format json`, session resume via `--continue`/`--conversation
<id>`, permission control via `--dangerously-skip-permissions` or a scoped
allowlist in ~/.gemini/antigravity-cli/settings.json. Structurally close
enough to Claude Code's `claude -p` that this file and claude_code_bridge.py
should stay easy to compare line-by-line — a real behavioral difference
between the two CLIs is worth a comment; an accidental divergence in how
this wrapper handles it is a bug.

Every public function returns the same shape as claude_code_bridge.py's
run_headless() and autobot/browser/extension_bridge.py's ExtensionBridge.run():
    {"ok": bool, "data": <dict | str | None>, "error": str}
and never raises.

No shell=True anywhere in this file: the prompt, cwd, and every other
caller-supplied value go straight into an argv list passed to subprocess,
never through a shell.

Safety, deliberately explicit: `skip_permissions` defaults to False — agy
respects whatever scoped allowlist is configured in
~/.gemini/antigravity-cli/settings.json (or asks and, since this is
headless, effectively can't act on anything not already allowlisted).
Callers (CoreLoop, via antigravity_tool.py) must opt in to
skip_permissions=True for "--dangerously-skip-permissions" behavior, and
CoreLoop's risk classifier (autobot/agent/approval.py) treats that opt-in
as DANGER-tier — the same treatment claude_code.run()'s
acceptEdits/bypassPermissions gets. This module does not make that safety
decision itself; it forwards whatever the caller asked for and reports
what actually happened.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

_DEFAULT_TIMEOUT = 600.0   # headless coding turns can run long, same rationale as claude_code_bridge.py

_CLI_NAME = "agy"


def is_available() -> bool:
    """True if the `agy` CLI is installed and on PATH."""
    return shutil.which(_CLI_NAME) is not None


def run_headless(
    prompt: str,
    cwd: str | None = None,
    skip_permissions: bool = False,
    model: str | None = None,
    effort: str | None = None,
    agent: str | None = None,
    continue_session: bool = False,
    conversation_id: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Run one headless Antigravity turn and return its result.

    skip_permissions:
      False (default) — agy respects the scoped allowlist in
                         ~/.gemini/antigravity-cli/settings.json. Anything
                         not pre-allowlisted simply won't happen in headless
                         mode (there's no one to answer an interactive
                         permission prompt), the same "the safe default
                         just declines rather than hanging" shape as Claude
                         Code's "plan" mode.
      True              — passes --dangerously-skip-permissions: agy
                           approves all tool calls, including file writes
                           and command execution, without asking. Most
                           capable, least safe — mirrors claude_code_bridge's
                           "bypassPermissions".

    conversation_id / continue_session let a caller resume a specific
    project's ongoing Antigravity conversation (e.g. "what's your status
    on the referral dashboard" as turn one, a follow-up prompt as turn two)
    instead of starting a fresh conversation with no memory of prior turns
    — this is the actual mechanism the orchestrator's per-project check-in
    loop depends on (see the project registry module).

    model / effort / agent forward directly to agy's own --model / --effort
    / --agent flags (see `agy models` / `agy agents` on the target machine
    for valid values) — left optional and unvalidated here on purpose,
    since agy's own accepted values can change out from under this file
    faster than this file should need updating to match.
    """
    if not prompt or not prompt.strip():
        return {"ok": False, "data": None, "error": "run_headless: prompt is required"}
    if not is_available():
        return {
            "ok": False,
            "data": None,
            "error": f"{_CLI_NAME} CLI not found on PATH. Install Google Antigravity "
                     f"and ensure its CLI is on PATH (see antigravity.google/docs/cli/).",
        }
    if cwd and not os.path.isdir(cwd):
        # subprocess.run(cwd=...) raises FileNotFoundError for a missing
        # directory and NotADirectoryError for a cwd that exists but is a
        # file — and the except FileNotFoundError clause below is there for
        # a *different* case (the agy binary itself vanishing from PATH
        # between the is_available() check above and exec). Without this
        # check, a bad cwd (e.g. a tracked project's working_dir that was
        # since moved or deleted — see project_registry.py) would raise
        # FileNotFoundError and get caught by that same clause, reporting
        # "agy CLI not found on PATH" — actively misleading a caller who
        # has agy installed just fine, since the real problem is unrelated
        # to the CLI at all.
        return {
            "ok": False,
            "data": None,
            "error": f"cwd does not exist or is not a directory: {cwd}",
        }

    args = [_CLI_NAME, "-p", prompt, "--output-format", "json"]
    if skip_permissions:
        args.append("--dangerously-skip-permissions")
    if model:
        args += ["--model", model]
    if effort:
        args += ["--effort", effort]
    if agent:
        args += ["--agent", agent]
    if conversation_id:
        args += ["--conversation", conversation_id]
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
        return {"ok": False, "data": None, "error": f"{_CLI_NAME} -p timed out after {timeout}s"}
    except FileNotFoundError:
        return {"ok": False, "data": None, "error": f"{_CLI_NAME} CLI not found on PATH."}
    except Exception as e:
        return {"ok": False, "data": None, "error": f"{type(e).__name__}: {e}"}

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    # agy's documented exit codes: 0 success, 1 model/config error, 2
    # streaming-protocol violation or unsupported slash command. None of
    # those are this function's problem to disambiguate beyond "ok: False,
    # here's stderr" — the caller (antigravity_tool.py) surfaces it as a
    # RuntimeError, same as claude_code_bridge's non-zero-exit path.
    if proc.returncode != 0:
        return {"ok": False, "data": out or None, "error": err or f"{_CLI_NAME} exited with code {proc.returncode}"}

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return {"ok": True, "data": {"result": out}, "error": "warning: could not parse JSON output"}

    # The docs' JSON envelope carries a `status` field independent of the
    # process exit code (SUCCESS, ERROR, CANCELED, INTERRUPTED, INVALID,
    # WAITING, RUNNING) — a clean exit (0) with status != SUCCESS is a real,
    # distinct outcome (e.g. the turn was interrupted or is still WAITING on
    # something) that a caller checking only proc.returncode would miss.
    status = parsed.get("status") if isinstance(parsed, dict) else None
    if status and status != "SUCCESS":
        return {"ok": False, "data": parsed, "error": f"{_CLI_NAME} status: {status}"}

    return {"ok": True, "data": parsed, "error": ""}
