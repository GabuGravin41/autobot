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
import uuid
from pathlib import Path
from typing import Any

from autobot.integrations.cli_exec import classify_error, is_batch_shim, resolve_exe, run_cli, strip_flag, unknown_flag

_DEFAULT_TIMEOUT = 600.0   # headless coding turns can run long, same rationale as claude_code_bridge.py

_CLI_NAME = "agy"

# A prompt longer than this, or containing characters cmd.exe mangles, is
# handed to agy through a file when agy is a .cmd/.bat shim (see cli_exec.py).
_MAX_INLINE_PROMPT = 6000
_CMD_UNSAFE = ("\n", "\r", "%", "!")


def is_available() -> bool:
    """True if the `agy` CLI is installed and on PATH."""
    return shutil.which(_CLI_NAME) is not None


def _timeout_flag(timeout: float) -> list[str]:
    # agy's own --print-timeout defaults to 5m and kills the run itself, so
    # any headless coding turn longer than five minutes died regardless of
    # the timeout this wrapper was given. Pass ours through (whole minutes,
    # the format agy documents: "5m").
    minutes = max(1, int((timeout + 59) // 60))
    return ["--print-timeout", f"{minutes}m"]


def _exclude_from_git(root: Path, pattern: str) -> None:
    """Add pattern to .git/info/exclude (local-only ignore; nothing tracked changes)."""
    exclude = root / ".git" / "info" / "exclude"
    try:
        if exclude.parent.is_dir():
            existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
            if pattern not in existing.splitlines():
                with exclude.open("a", encoding="utf-8") as fh:
                    fh.write(("" if existing.endswith("\n") or not existing else "\n") + pattern + "\n")
    except OSError:
        pass


def _prompt_via_file(prompt: str, cwd: str | None) -> tuple[str, Path]:
    """Write a long/multi-line prompt to a file inside the workspace and
    return a short one-line prompt that points at it."""
    base = Path(cwd) if cwd else Path.home() / ".autobot"
    folder = base / ".autobot"
    folder.mkdir(parents=True, exist_ok=True)
    if cwd:
        _exclude_from_git(Path(cwd), ".autobot/")
    path = folder / f"task_{uuid.uuid4().hex[:10]}.md"
    path.write_text(prompt, encoding="utf-8")
    short = (f"Your full instructions are in the file .autobot/{path.name} in the current "
             f"workspace. Read that file first and carry out its instructions exactly.")
    return short, path


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
    *,
    json_schema: dict | str | None = None,
) -> dict[str, Any]:
    """
    Run one headless Antigravity turn and return
    {"ok", "data", "error", "error_class"}. Never raises.

    skip_permissions=False (default) respects the allowlist in
    ~/.gemini/antigravity-cli/settings.json; True passes
    --dangerously-skip-permissions. json_schema -> data["structured_output"]
    (documented at antigravity.google/docs/cli/headless/).
    """
    if not prompt or not prompt.strip():
        return {"ok": False, "data": None, "error": "run_headless: prompt is required", "error_class": "other"}
    if not is_available():
        return {
            "ok": False, "data": None, "error_class": "not_installed",
            "error": f"{_CLI_NAME} CLI not found on PATH. Install Google Antigravity "
                     f"and ensure its CLI is on PATH (see antigravity.google/docs/cli/).",
        }
    if cwd and not os.path.isdir(cwd):
        return {"ok": False, "data": None, "error_class": "other",
                "error": f"cwd does not exist or is not a directory: {cwd}"}

    prompt_arg = prompt
    prompt_file: Path | None = None
    exe = resolve_exe(_CLI_NAME)
    if is_batch_shim(exe) and (len(prompt) > _MAX_INLINE_PROMPT or any(c in prompt for c in _CMD_UNSAFE)):
        prompt_arg, prompt_file = _prompt_via_file(prompt, cwd)

    schema_file: Path | None = None
    extra: list[str] = []
    if json_schema is not None:
        tmp_dir = Path.home() / ".autobot" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        schema_file = tmp_dir / f"schema_{uuid.uuid4().hex[:10]}.json"
        schema_file.write_text(json_schema if isinstance(json_schema, str) else json.dumps(json_schema),
                               encoding="utf-8")
        extra += ["--json-schema", str(schema_file)]

    def build(with_timeout_flag: bool) -> list[str]:
        args = [_CLI_NAME, "-p", prompt_arg, "--output-format", "json"]
        if with_timeout_flag:
            args += _timeout_flag(timeout)
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
        return args + extra

    try:
        argv = build(True)
        res = run_cli(argv, cwd=cwd, timeout=timeout + 30)
        for _ in range(4):
            if res.ok or res.timed_out or res.not_found:
                break
            flag = unknown_flag(res.stderr + " " + res.stdout)
            if not flag or flag not in argv or flag in ("--output-format",):
                break
            argv = strip_flag(argv, flag, {"--print-timeout", "--model", "--effort", "--agent",
                                           "--conversation", "--json-schema", "--output-format"})
            res = run_cli(argv, cwd=cwd, timeout=timeout + 30)
    finally:
        for f in (prompt_file, schema_file):
            if f is not None:
                try:
                    f.unlink()
                except OSError:
                    pass

    if res.not_found:
        return {"ok": False, "data": None, "error": res.stderr, "error_class": "not_installed"}
    if res.timed_out:
        return {"ok": False, "data": res.stdout or None, "error": f"{_CLI_NAME} -p timed out after {timeout}s",
                "error_class": "timeout"}

    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()

    if res.returncode != 0:
        return {"ok": False, "data": out or None,
                "error": err or f"{_CLI_NAME} exited with code {res.returncode}",
                "error_class": classify_error(f"{err} {out}")}

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return {"ok": True, "data": {"result": out, "response": out},
                "error": "warning: could not parse JSON output", "error_class": ""}

    status = parsed.get("status") if isinstance(parsed, dict) else None
    if status and status != "SUCCESS":
        detail = str(parsed.get("error") or "").strip()
        message = f"{_CLI_NAME} status: {status}" + (f": {detail}" if detail else "")
        return {"ok": False, "data": parsed, "error": message, "error_class": classify_error(message)}

    return {"ok": True, "data": parsed, "error": "", "error_class": ""}
