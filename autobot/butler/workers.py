"""
Workers: the capable tools the butler hands real work to, the way you do.

  claude_code   Claude Code headless (`claude -p`) — your Claude subscription
  antigravity   Antigravity headless (`agy -p`) — your Antigravity subscription
  vscode        not an agent on its own (no headless mode); the butler can
                open a folder in VS Code for YOU to review, not work in it

Every worker call ends with a structured REPORT (JSON schema enforced by
the CLI's --json-schema) so the butler never has to scrape prose for
"did it push a kernel / does it need input". The report is informative,
not authoritative: completion is still decided by the task's checks.

Safety: write-capable runs get hard denials for the irreversible things the
butler must do itself, only after your approval — pushing code, submitting
to Kaggle, deleting trees. Claude Code enforces these via --disallowedTools.
Antigravity has no equivalent flag, so for agy the same rules are stated in
the instructions and the task's approval floor still applies to anything
the BUTLER does; say so plainly rather than pretend it's enforced.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

from autobot.util.jsonx import extract_json

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["done", "in_progress", "needs_input", "blocked"]},
        "summary": {"type": "string", "description": "What you did and what the result is, in plain words."},
        "question": {"type": "string", "description": "Only if status is needs_input/blocked: the one question for the user."},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "kernels_pushed": {"type": "array", "items": {"type": "string"},
                           "description": "Kaggle kernel slugs (owner/slug) you pushed this turn."},
        "next_step": {"type": "string", "description": "What should happen next, if anything."},
        "submission_candidate": {
            "type": "object",
            "description": "Only when a Kaggle submission is ready for the user's approval.",
            "properties": {
                "competition": {"type": "string"}, "file": {"type": "string"},
                "kernel": {"type": "string"}, "version": {"type": "integer"}, "message": {"type": "string"},
            },
        },
    },
    "required": ["status", "summary"],
}

# Irreversible or outward-facing actions workers must never take themselves.
HARD_DENY_CLAUDE = [
    "Bash(git push:*)", "Bash(git push *)",
    "Bash(kaggle competitions submit:*)", "Bash(kaggle competitions submit *)",
    "Bash(kaggle c submit *)",
    "Bash(rm -rf *)", "Bash(rm -r *)", "Bash(rmdir /s *)", "Bash(del /s *)",
    "Bash(Remove-Item * -Recurse*)", "Bash(format *)", "Bash(shutdown *)",
]

# Commands a write-capable Claude worker may run without asking.
DEV_ALLOW_CLAUDE = [
    "Read", "Edit", "Write", "MultiEdit", "Glob", "Grep", "NotebookEdit", "TodoWrite",
    "Bash(python *)", "Bash(py *)", "Bash(python3 *)", "Bash(pip install *)", "Bash(pytest *)",
    "Bash(npm test*)", "Bash(npm run *)", "Bash(npm install*)", "Bash(npx *)", "Bash(node *)",
    "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)", "Bash(git add *)", "Bash(git commit *)",
    "Bash(git checkout -b *)", "Bash(git switch *)", "Bash(git branch*)",
    "Bash(latexmk *)", "Bash(pdflatex *)", "Bash(xelatex *)", "Bash(bibtex *)", "Bash(biber *)",
    "Bash(ls*)", "Bash(dir*)", "Bash(cat *)", "Bash(type *)", "Bash(mkdir *)",
]
KAGGLE_ALLOW_CLAUDE = [
    "Bash(kaggle kernels *)", "Bash(kaggle competitions list*)", "Bash(kaggle competitions files*)",
    "Bash(kaggle competitions download*)", "Bash(kaggle competitions leaderboard*)",
    "Bash(kaggle competitions submissions*)", "Bash(kaggle datasets *)", "Bash(kaggle models *)",
    "Bash(autobot kaggle *)", "Bash(python -m autobot kaggle *)",
]
RESEARCH_ALLOW_CLAUDE = ["WebSearch", "WebFetch"]

WORKERS = ("claude_code", "antigravity", "vscode")


@dataclass
class WorkerPolicy:
    writes: bool = True                 # False = read-only (plan mode / no skip-permissions)
    allowed: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=lambda: list(HARD_DENY_CLAUDE))
    agy_skip_permissions: bool = True   # agy can't do anything headless without it unless allowlisted
    model: str | None = None
    timeout: float = 1800.0


@dataclass
class WorkerResult:
    ok: bool
    worker: str
    text: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    error: str = ""
    error_class: str = ""   # rate_limited | auth | timeout | not_installed | unsupported | other
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "worker": self.worker, "text": self.text[:4000], "report": self.report,
                "session_id": self.session_id, "error": self.error[:2000], "error_class": self.error_class}


def available_workers() -> list[str]:
    out = []
    if shutil.which("claude"):
        out.append("claude_code")
    if shutil.which("agy"):
        out.append("antigravity")
    return out


def pick_worker(requested: str | None) -> str | None:
    """Honor an explicit choice; 'auto'/None -> first available (Claude Code, then Antigravity)."""
    if requested and requested not in ("auto", ""):
        return requested
    avail = available_workers()
    return avail[0] if avail else None


def _report_from(data: Any, text: str) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("structured_output"), dict):
        return data["structured_output"]
    parsed = extract_json(text)
    if isinstance(parsed, dict) and "summary" in parsed:
        return parsed
    return {"status": "unknown", "summary": text[:2000]}


def run_worker(
    worker: str,
    instruction: str,
    cwd: str | None,
    policy: WorkerPolicy,
    session_id: str | None = None,
) -> WorkerResult:
    """Run one turn of the chosen worker. Never raises."""
    if cwd and not os.path.isdir(cwd):
        return WorkerResult(False, worker, error=f"project folder does not exist: {cwd}", error_class="other")

    if worker == "claude_code":
        from autobot.integrations import claude_code_bridge as b
        r = b.run_headless(
            instruction, cwd=cwd,
            permission_mode="acceptEdits" if policy.writes else "plan",
            allowed_tools=policy.allowed or None,
            disallowed_tools=policy.denied or None,
            resume_session_id=session_id,
            json_schema=REPORT_SCHEMA,
            model=policy.model,
            timeout=policy.timeout,
        )
        data = r.get("data")
        text = str(data.get("result", "")) if isinstance(data, dict) else str(data or "")
        return WorkerResult(
            ok=bool(r.get("ok")), worker=worker, text=text, report=_report_from(data, text) if r.get("ok") else {},
            session_id=(data.get("session_id") if isinstance(data, dict) else None) or session_id,
            error=r.get("error", "") if not r.get("ok") else "", error_class=r.get("error_class", ""), raw=data,
        )

    if worker == "antigravity":
        from autobot.integrations import antigravity_bridge as b
        r = b.run_headless(
            instruction, cwd=cwd,
            skip_permissions=policy.writes and policy.agy_skip_permissions,
            model=policy.model, conversation_id=session_id,
            json_schema=REPORT_SCHEMA, timeout=policy.timeout,
        )
        data = r.get("data")
        text = str(data.get("response") or data.get("result") or "") if isinstance(data, dict) else str(data or "")
        return WorkerResult(
            ok=bool(r.get("ok")), worker=worker, text=text, report=_report_from(data, text) if r.get("ok") else {},
            session_id=(data.get("conversation_id") if isinstance(data, dict) else None) or session_id,
            error=r.get("error", "") if not r.get("ok") else "", error_class=r.get("error_class", ""), raw=data,
        )

    if worker == "vscode":
        return WorkerResult(
            False, worker, error_class="unsupported",
            error=("VS Code has no headless agent mode Autobot can drive reliably. Use claude_code or "
                   "antigravity for the work; the butler can open the folder in VS Code for your review."),
        )

    return WorkerResult(False, worker, error=f"unknown worker {worker!r}", error_class="unsupported")


def open_in_vscode(path: str) -> bool:
    """Open a folder in VS Code for the user (review), if the `code` CLI exists."""
    from autobot.integrations.cli_exec import run_cli
    if not shutil.which("code"):
        return False
    return run_cli(["code", path], timeout=30).ok
