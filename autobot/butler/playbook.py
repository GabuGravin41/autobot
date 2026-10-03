"""
The playbook: how one task moves from "Dalton wants X" to done.

This is a state machine, not a free-form LLM loop. Each call to step()
performs exactly one phase transition and leaves the task in a resting
status (queued / waiting / needs_you / done / failed). The phases:

  start     resolve the worker and the working folder
  propose   (only if the user gave no checks) the worker proposes acceptance
            checks, read-only; unsafe proposals are dropped
  work      one worker turn: first the full brief, afterwards feedback,
            resuming the worker's own session so it keeps its context
  wait      Kaggle only: kernels are running; the daemon polls, the worker sleeps
  verify    the daemon runs the checks itself; the worker's word is not enough
  review    no checks exist: Dalton decides from the inbox
  escalate  attempts exhausted: optional LLM triage, otherwise ask Dalton
  finalize  write the summary, notify, done

Where an LLM "manager" is used at all (escalation triage), it chooses among
a few fixed moves with a JSON schema; if none is configured, the playbook
asks you instead. Nothing here lets a model declare a task done.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from autobot.butler import checks as checks_mod
from autobot.butler.lanes import COMMON_RULES, LANES, LaneSpec
from autobot.butler.store import (
    DONE,
    FAILED,
    NEEDS_YOU,
    PAUSED,
    QUEUED,
    WAITING,
    ButlerStore,
    Task,
)
from autobot.butler.workers import HARD_DENY_CLAUDE, WorkerPolicy, WorkerResult, pick_worker, run_worker

logger = logging.getLogger(__name__)

CRITERIA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": sorted(checks_mod.VALID_TYPES)},
                    "run": {"type": "string"}, "path": {"type": "string"}, "main": {"type": "string"},
                    "patterns": {"type": "array", "items": {"type": "string"}},
                    "pattern": {"type": "string"}, "min": {"type": "integer"},
                    "expect_exit": {"type": "integer"}, "contains": {"type": "string"},
                },
                "required": ["type"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["criteria"],
}

ESCALATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "move": {"type": "string", "enum": ["retry_with_new_approach", "switch_worker", "ask_user"]},
        "instruction": {"type": "string", "description": "For retry: the concrete new approach to tell the worker."},
        "question": {"type": "string", "description": "For ask_user: one specific question for Dalton."},
        "reason": {"type": "string"},
    },
    "required": ["move", "reason"],
}

RATE_LIMIT_COOLDOWN_S = float(os.getenv("AUTOBOT_RATE_LIMIT_COOLDOWN_MIN", "45")) * 60
KERNEL_POLL_S = float(os.getenv("AUTOBOT_KERNEL_POLL_MIN", "3")) * 60


def _cfg(task: Task, key: str, default: Any) -> Any:
    v = task.config.get(key, default)
    return default if v is None else v


class Playbook:
    def __init__(self, store: ButlerStore, llm: Any | None = None, worker_fn=run_worker, check_fn=None,
                 kaggle_status_fn=None, gmail=None) -> None:
        self.store = store
        if gmail is None:
            from autobot.integrations import gmail as gmail_mod
            gmail = gmail_mod
        self.gmail = gmail
        self.llm = llm
        self.worker_fn = worker_fn
        self.check_fn = check_fn or checks_mod.run_checks
        self.kaggle_status_fn = kaggle_status_fn or _kaggle_statuses

    # ── entry point ─────────────────────────────────────────────────────────

    def step(self, task: Task) -> None:
        """Advance one phase. Always leaves the task in a resting status."""
        phase = task.state.get("phase", "start")
        handler = getattr(self, f"_phase_{phase}", None)
        if handler is None:
            self._fail(task, f"unknown phase {phase!r}")
            return
        handler(task)

    # ── helpers ─────────────────────────────────────────────────────────────

    def lane(self, task: Task) -> LaneSpec:
        return LANES.get(task.lane) or LANES["coding"]

    def _set(self, task: Task, status: str, wake_in: float = 0.0, **state: Any) -> None:
        if state:
            self.store.merge_state(task.id, **state)
        self.store.update_task(task.id, status=status, next_wake_at=time.time() + wake_in)

    def _next(self, task: Task, phase: str, wake_in: float = 0.0, **state: Any) -> None:
        self._set(task, QUEUED if wake_in <= 0 else WAITING, wake_in, phase=phase, **state)

    def _fail(self, task: Task, reason: str) -> None:
        self.store.log(task.id, "failed", reason)
        self.store.update_task(task.id, status=FAILED, result=reason)

    def _ask(self, task: Task, question: str, kind: str = "question", payload: dict | None = None,
             resume_phase: str = "work") -> None:
        self.store.add_approval(task.id, kind, question, payload or {})
        self._set(task, NEEDS_YOU, phase="awaiting_user", resume_phase=resume_phase)

    def _policy(self, task: Task, lane: LaneSpec, writes: bool | None = None) -> WorkerPolicy:
        allowed = list(lane.allowed_tools)
        # Let the worker run the exact acceptance commands itself (and this
        # interpreter by full path). Without this, "Bash(python *)" did not
        # match "/usr/bin/python -m unittest" and the worker in the first real
        # end-to-end run had to verify its fix by hand.
        for c in task.criteria or []:
            if c.get("type") == "command" and c.get("run"):
                allowed += [f"Bash({c['run']})", f"Bash({c['run']} *)"]
        exe = sys.executable
        allowed += [f"Bash({exe} *)", f'Bash("{exe}" *)']
        denied = list(HARD_DENY_CLAUDE)
        if lane.name == "email":
            # Reading untrusted mail + shell or web access = a way to exfiltrate or act on
            # injected instructions. Triage gets files only; compose may search the web but
            # never sees the inbox.
            allowed = list(lane.allowed_tools)
            denied += ["Bash", "WebFetch"]
            if _cfg(task, "mode", "triage") == "compose" and _cfg(task, "allow_web", True):
                allowed += ["WebSearch"]
            else:
                denied += ["WebSearch"]
        return WorkerPolicy(
            writes=_cfg(task, "writes", True) if writes is None else writes,
            allowed=allowed,
            denied=denied,
            agy_skip_permissions=_cfg(task, "agy_skip_permissions", True),
            model=task.config.get("model"),
            timeout=float(_cfg(task, "worker_timeout_min", 30)) * 60,
        )

    def _cwd(self, task: Task) -> str | None:
        return task.state.get("cwd") or task.project_dir

    # ── phases ──────────────────────────────────────────────────────────────

    def _phase_start(self, task: Task) -> None:
        lane = self.lane(task)
        worker = pick_worker(task.worker)
        if worker is None:
            self._ask(task, "No coding worker is available: neither `claude` nor `agy` was found on PATH. "
                            "Install/sign in to one of them, then approve this item to retry.",
                      resume_phase="start")
            return
        if worker == "vscode":
            self._ask(task, "VS Code can't be driven headlessly. Approve to run this task with Claude Code "
                            "instead (or reject and change the task's worker to antigravity).",
                      payload={"switch_to": "claude_code"}, resume_phase="start")
            return

        cwd = task.project_dir
        if lane.needs_project_dir:
            if not cwd:
                self._fail(task, f"Lane '{lane.name}' needs a project folder (--dir).")
                return
            if not os.path.isdir(cwd):
                self._fail(task, f"Project folder does not exist: {cwd}")
                return
        else:
            if not cwd:
                from autobot.paths import workspaces_dir
                cwd = str(workspaces_dir() / f"task-{task.id}")
            os.makedirs(cwd, exist_ok=True)

        if worker != task.worker:
            self.store.update_task(task.id, worker=worker)
        branch_note = self._maybe_branch(task, cwd) if lane.name == "coding" and _cfg(task, "branch", True) else None
        self.store.log(task.id, "started", f"Worker: {worker}. Folder: {cwd}" + (f". {branch_note}" if branch_note else ""))

        if lane.name == "email":
            if worker != "claude_code":
                from autobot.butler.workers import available_workers
                if "claude_code" not in available_workers():
                    self._ask(task, "The email lane needs Claude Code (it's the only worker whose tools Autobot can "
                                    "lock down to 'read and write files only'). Install/sign in to `claude`, then approve.",
                              resume_phase="start")
                    return
                worker = "claude_code"
                self.store.update_task(task.id, worker=worker)
            if not self.gmail.is_configured():
                self._ask(task, "Gmail isn't connected yet. Follow GMAIL_SETUP.md, run `autobot butler gmail-auth`, "
                                "then approve this item.", resume_phase="start")
                return
            mode = _cfg(task, "mode", "triage")
            self._next(task, "sync" if mode == "triage" else "work", cwd=cwd, worker=worker, round=1)
            return

        criteria = task.criteria or lane.default_checks(task.config)
        if criteria and not task.criteria:
            self.store.update_task(task.id, criteria=criteria, criteria_source="lane_default")
        if not criteria and lane.propose_criteria and _cfg(task, "propose_criteria", True):
            self._next(task, "propose", cwd=cwd, worker=worker, round=1)
        else:
            self._next(task, "work", cwd=cwd, worker=worker, round=1)

    def _phase_propose(self, task: Task) -> None:
        lane = self.lane(task)
        worker = task.state.get("worker") or task.worker or "claude_code"
        if self._cooling(task, worker):
            return
        prompt = f"""You are helping Autobot (Dalton's task manager) define how to CHECK a task automatically.
Do not change any files. Inspect the project folder and propose acceptance checks.

TASK: {task.title}
WHAT DALTON WANTS (his words):
{task.intent}

Propose 1-5 checks that would prove the task is done. Allowed check types:
- command: {{"type":"command","run":"pytest -q tests/test_x.py","expect_exit":0}}  (test runners/compilers only:
  pytest, python -m pytest, npm test, npm run <script>, cargo test, go test, latexmk, tsc, ruff, mypy...;
  no shell operators like && | >)
- file_exists: {{"type":"file_exists","path":"relative/path"}}
- csv_rows: {{"type":"csv_rows","path":"out.csv","min":1}}
- latex_compiles: {{"type":"latex_compiles","main":"main.tex"}}
- contains: {{"type":"contains","path":"file","patterns":["must appear"]}}
Prefer checks that exist or will exist in the project (e.g. an existing test suite, or a new test
the task should add). Reply with the JSON object only."""
        res = self.worker_fn(worker, prompt, self._cwd(task), self._policy(task, lane, writes=False))
        if not res.ok:
            if self._handle_worker_failure(task, res, retry_phase="propose"):
                return
            self.store.log(task.id, "propose_failed", f"Could not get proposed checks: {res.error[:300]}")
            self._next(task, "work")
            return
        proposed = res.report.get("criteria") if isinstance(res.report, dict) else None
        if proposed is None:
            from autobot.util.jsonx import extract_json
            data = extract_json(res.text) or {}
            proposed = data.get("criteria") if isinstance(data, dict) else None
        kept, dropped = [], []
        for c in proposed or []:
            why = checks_mod.validate(c, proposed=True)
            (dropped if why else kept).append((c, why))
        criteria = [c for c, _ in kept]
        self.store.update_task(task.id, criteria=criteria, criteria_source="proposed" if criteria else "none")
        self.store.log(task.id, "criteria", f"{len(criteria)} checks proposed by {worker}"
                       + (f"; {len(dropped)} dropped as unsafe/invalid" if dropped else ""),
                       {"kept": criteria, "dropped": [{"check": c, "why": w} for c, w in dropped]})
        self._next(task, "work", session=None)

    def _phase_work(self, task: Task) -> None:
        lane = self.lane(task)
        worker = task.state.get("worker") or task.worker or "claude_code"
        if self._cooling(task, worker):
            return
        feedback = task.state.get("feedback")
        session = task.state.get("session")
        if session and feedback:
            instruction = self._followup(task, feedback)
        elif feedback:
            # No session to resume (older CLI output, or a fresh worker): the
            # feedback must still reach the worker, appended to the full brief.
            instruction = self._brief(task, lane) + f"\n\nLATEST FEEDBACK (from earlier attempts)\n{feedback}"
        else:
            instruction = self._brief(task, lane)
        started = time.time()
        self.store.log(task.id, "worker_start",
                       f"{worker}: attempt {task.attempts + 1}" + (" (continuing its session)" if session and feedback else ""))

        res = self.worker_fn(worker, instruction, self._cwd(task), self._policy(task, lane), session)
        if not res.ok and session and re.search(r"session|conversation", res.error, re.I):
            # The worker's session expired or can't be resumed: start fresh with the full brief.
            self.store.log(task.id, "session_reset", "Could not resume the worker's session; starting a fresh one.")
            instruction = self._brief(task, lane) + (f"\n\nLATEST FEEDBACK\n{feedback}" if feedback else "")
            res = self.worker_fn(worker, instruction, self._cwd(task), self._policy(task, lane), None)

        if not res.ok:
            if self._handle_worker_failure(task, res, retry_phase="work"):
                return
            attempts = task.attempts + 1
            self.store.update_task(task.id, attempts=attempts)
            self.store.log(task.id, "worker_error", f"{worker} failed: {res.error[:500]}", res.to_dict())
            if attempts >= int(_cfg(task, "max_attempts", 3)):
                self._next(task, "escalate", last_failure=res.error[:1500])
            else:
                self._next(task, "work", wake_in=60, feedback=f"Your previous turn failed with: {res.error[:800]}. Continue the task.")
            return

        report = res.report or {}
        self.store.merge_state(task.id, session=res.session_id, last_report=report, feedback=None)
        self.store.log(task.id, "worker_report", f"{worker} ({report.get('status', '?')}): {str(report.get('summary', res.text))[:600]}",
                       {"report": report, "seconds": round(time.time() - started)})

        if report.get("status") in ("needs_input", "blocked") and report.get("question"):
            self._ask(task, f"{task.title}: {report['question']}", resume_phase="work")
            return

        if lane.rounds:
            self._after_kaggle_round(task, report, started)
            return
        if lane.name == "email":
            self._next(task, "publish")
            return
        self._next(task, "verify")

    def _phase_verify(self, task: Task) -> None:
        task = self.store.get_task(task.id) or task
        criteria = task.criteria
        if not criteria:
            summary = (task.state.get("last_report") or {}).get("summary", "")
            self.store.add_approval(task.id, "review",
                                    f"{task.title}: no automatic checks exist, so this needs your review. "
                                    f"Approve = accept as done; reject with a note = send it back.",
                                    {"summary": summary, "cwd": self._cwd(task)})
            self._set(task, NEEDS_YOU, phase="awaiting_user", resume_phase="work")
            return
        results = self.check_fn(criteria, self._cwd(task) or ".")
        summary = checks_mod.summarize(results)
        passed = all(r.passed for r in results)
        self.store.log(task.id, "checks", ("All checks passed" if passed else "Checks failed") + "\n" + summary,
                       {"results": [r.to_dict() for r in results]})
        if passed:
            self._next(task, "finalize", check_summary=summary)
            return
        attempts = task.attempts + 1
        self.store.update_task(task.id, attempts=attempts)
        if attempts >= int(_cfg(task, "max_attempts", 3)):
            self._next(task, "escalate", last_failure=summary)
        else:
            self._next(task, "work", feedback=(
                "Autobot ran the acceptance checks after your last turn. Results:\n" + summary +
                "\n\nFix the failures (don't weaken the checks), then report again."))

    def _phase_escalate(self, task: Task) -> None:
        failure = task.state.get("last_failure", "")
        escalations = int(task.state.get("escalations", 0))
        decision: dict[str, Any] = {"move": "ask_user", "reason": "no manager model configured"}
        if self.llm is not None and escalations < int(_cfg(task, "max_escalations", 1)):
            try:
                decision = self.llm.complete_json(
                    "You are the manager of an automated work queue. A worker (an AI coding agent) has "
                    "failed a task several times. Pick ONE move. Prefer ask_user when the failure needs "
                    "information, credentials, or a judgment only the owner can make.",
                    f"TASK: {task.title}\nINTENT: {task.intent}\nWORKER: {task.state.get('worker')}\n"
                    f"ATTEMPTS: {task.attempts}\nLAST FAILURE:\n{failure[:3000]}\n",
                    ESCALATION_SCHEMA,
                )
            except Exception as e:
                self.store.log(task.id, "manager_error", f"Manager model unavailable: {e}")
        move = decision.get("move")
        self.store.log(task.id, "escalation", f"{move}: {decision.get('reason', '')}", decision)
        if move == "retry_with_new_approach" and decision.get("instruction"):
            self.store.update_task(task.id, attempts=0)
            self._next(task, "work", escalations=escalations + 1,
                       feedback=f"Previous attempts failed:\n{failure[:1500]}\n\nTry this approach instead: {decision['instruction']}")
            return
        if move == "switch_worker":
            current = task.state.get("worker")
            other = "antigravity" if current == "claude_code" else "claude_code"
            from autobot.butler.workers import available_workers
            if other in available_workers():
                self.store.update_task(task.id, attempts=0, worker=other)
                self._next(task, "work", escalations=escalations + 1, worker=other, session=None,
                           feedback=f"Another worker tried this and failed:\n{failure[:1500]}")
                return
        question = decision.get("question") or (
            f"{task.title}: the worker failed {task.attempts} time(s). Last problem:\n{failure[:1200]}\n\n"
            "Reply with guidance and approve to retry, or reject to stop this task.")
        self._ask(task, question, resume_phase="work")

    def _phase_finalize(self, task: Task, notify: bool = True) -> None:
        report = task.state.get("last_report") or {}
        parts = []
        if task.state.get("check_summary"):
            parts.append("Checks:\n" + task.state["check_summary"])
        parts.append(str(report.get("summary") or "(no summary)"))
        diff = self._diffstat(task)
        if diff:
            parts.append("Changes:\n" + diff)
        summary = "\n\n".join(parts)
        self.store.update_task(task.id, status=DONE, result=summary[:8000])
        self.store.merge_state(task.id, phase="done")
        if notify:
            self.store.add_approval(task.id, "done", f"Done: {task.title}", {"summary": summary[:8000], "cwd": self._cwd(task)})
        self.store.log(task.id, "done", "Task finished.")
        if task.lane == "coding" and task.config.get("open_vscode") and self._cwd(task):
            from autobot.butler.workers import open_in_vscode
            open_in_vscode(self._cwd(task))

    def _phase_sync(self, task: Task) -> None:
        cwd = Path(self._cwd(task) or ".")
        stamp = time.strftime("%Y-%m-%d")
        try:
            new = self.gmail.sync_recent(cwd / "mail" / stamp,
                                         query=_cfg(task, "query", "newer_than:2d -category:promotions -category:social"),
                                         max_messages=int(_cfg(task, "max_messages", 60)))
        except Exception as e:
            self.store.log(task.id, "gmail_error", f"Gmail sync failed: {e}")
            if "gmail-auth" in str(e) or "not connected" in str(e).lower():
                self._ask(task, f"{e}", resume_phase="sync")
            else:
                self._set(task, WAITING, 900, phase="sync")
            return
        self.store.log(task.id, "gmail_sync", f"{len(new)} new message(s)")
        if not new:
            self._finalize_run(task, "No new mail.", notify=False)
            return
        rel = [str(p.relative_to(cwd)).replace("\\", "/") for p in new]
        self._next(task, "work", new_mail=rel, digest_path=f"digests/{stamp}-{time.strftime('%H%M')}.md",
                   session=None, feedback=None)

    def _phase_publish(self, task: Task) -> None:
        cwd = Path(self._cwd(task) or ".")
        published = set(task.state.get("published") or [])
        errors, created = [], []
        for f in sorted((cwd / "drafts").glob("*.md")):
            key = f.name
            if key in published:
                continue
            try:
                d = self.gmail.parse_draft_file(f)
                ref = self.gmail.create_draft(d)
            except ValueError as e:
                errors.append(str(e))
                continue
            except Exception as e:
                self.store.log(task.id, "gmail_error", f"Creating draft {key} failed: {e}")
                errors.append(f"{key}: Gmail error {e}")
                continue
            published.add(key)
            created.append(d["subject"])
            self.store.add_approval(task.id, "send_email", f"Send email to {d['to']}: {d['subject']}",
                                    {"draft_id": ref["draft_id"], "to": d["to"], "cc": d.get("cc", ""),
                                     "subject": d["subject"], "summary": d["body"][:4000], "file": str(f)})
        self.store.merge_state(task.id, published=sorted(published))
        if created:
            self.store.log(task.id, "drafts", f"{len(created)} draft(s) created in Gmail, waiting for your approval to send")
        bad_format = [e for e in errors if "Gmail error" not in e]
        if bad_format and task.attempts + 1 < int(_cfg(task, "max_attempts", 3)):
            self.store.update_task(task.id, attempts=task.attempts + 1)
            self._next(task, "work", feedback="Some draft files could not be used:\n" + "\n".join(bad_format)
                       + "\nFix those files (see DRAFT FILE FORMAT) and report again.")
            return
        digest = task.state.get("digest_path")
        text = ""
        if digest and (cwd / digest).is_file():
            text = (cwd / digest).read_text(encoding="utf-8", errors="replace")
            self.store.add_approval(task.id, "digest", f"Mail digest ({len(task.state.get('new_mail') or [])} new)",
                                    {"summary": text[:6000], "file": str(cwd / digest)})
        summary = f"{len(created)} draft(s) waiting for your approval." + (f" Digest: {digest}" if text else "")
        self._finalize_run(task, summary, notify=not text and not created)

    def _finalize_run(self, task: Task, summary: str, notify: bool) -> None:
        """End one run of a (possibly recurring) task."""
        every = task.config.get("every_minutes")
        if every:
            keep = {k: task.state.get(k) for k in ("cwd", "worker", "published") if task.state.get(k) is not None}
            self.store.update_task(task.id, attempts=0, state={**keep, "phase": task.state.get("first_phase", "sync"
                                   if task.lane == "email" else "work"), "last_run_summary": summary})
            self.store.update_task(task.id, status=WAITING, next_wake_at=time.time() + float(every) * 60)
            self.store.log(task.id, "run_done", f"{summary} Next run in {every} min.")
            return
        self.store.merge_state(task.id, last_report={**(task.state.get("last_report") or {}),
                                                     "summary": summary + "\n\n" + str((task.state.get("last_report") or {}).get("summary", ""))})
        task = self.store.get_task(task.id) or task
        self._phase_finalize(task, notify=notify)

    def _phase_wait(self, task: Task) -> None:
        kernels = task.state.get("kernels") or []
        statuses = self.kaggle_status_fn(kernels)
        pending = [k for k in kernels if statuses.get(k, {}).get("status") not in ("complete", "error", "cancelled")]
        waited = time.time() - float(task.state.get("wait_started", time.time()))
        if pending and waited < float(_cfg(task, "kernel_wait_hours", 12)) * 3600:
            self._set(task, WAITING, KERNEL_POLL_S, phase="wait")
            return
        lines = []
        for k in kernels:
            s = statuses.get(k, {})
            line = f"- {k}: {s.get('status', 'unknown')}"
            if s.get("error_log"):
                line += f" — {str(s['error_log'])[:500]}"
            lines.append(line)
        if pending:
            lines.append(f"(stopped waiting after {waited / 3600:.1f}h; still pending: {', '.join(pending)})")
        self.store.log(task.id, "kernels_finished", "\n".join(lines))
        rnd = int(task.state.get("round", 1))
        if rnd >= int(_cfg(task, "max_rounds", 4)):
            self._next(task, "finalize", check_summary="Kernel results:\n" + "\n".join(lines))
            return
        self._next(task, "work", round=rnd + 1, kernels=[], feedback=(
            "The kernels you pushed have finished:\n" + "\n".join(lines) +
            "\n\nDownload the outputs you need with `" + _kaggle_cli_cmd() + " output <slug> -p <dir>`, evaluate them "
            "honestly (compare CV to leaderboard, check the submission file), update THINKING_AND_DECISIONS.md, "
            f"then do round {rnd + 1}: the most promising next experiment."))

    def _phase_awaiting_user(self, task: Task) -> None:
        # Nothing to do until the user answers; the daemon routes decisions via on_decision().
        self._set(task, NEEDS_YOU)

    # ── Kaggle rounds ───────────────────────────────────────────────────────

    def _after_kaggle_round(self, task: Task, report: dict, started: float) -> None:
        kernels = [k for k in (report.get("kernels_pushed") or []) if isinstance(k, str) and "/" in k]
        kernels += [k for k in _ledger_kernels_since(started, self._cwd(task)) if k not in kernels]
        cand = report.get("submission_candidate")
        if isinstance(cand, dict) and (cand.get("file") or cand.get("kernel")):
            comp = cand.get("competition") or task.config.get("competition")
            self.store.add_approval(task.id, "kaggle_submit",
                                    f"Submit to Kaggle '{comp}'? {cand.get('message', '')}".strip(),
                                    {**cand, "competition": comp, "cwd": self._cwd(task)})
        rnd = int(task.state.get("round", 1))
        if kernels:
            self.store.log(task.id, "waiting", f"Round {rnd}: waiting for {len(kernels)} kernel(s): {', '.join(kernels)}")
            self._set(task, WAITING, KERNEL_POLL_S, phase="wait", kernels=kernels, wait_started=time.time())
            return
        if report.get("status") == "done" or rnd >= int(_cfg(task, "max_rounds", 4)):
            self._next(task, "finalize")
            return
        self._next(task, "work", round=rnd + 1, feedback=(
            f"Round {rnd} ended without pushing a kernel. Continue: push the next experiment with "
            f"`{_kaggle_cli_cmd()} push <dir>`, or report status done if the competition work is complete."))

    # ── decisions from the inbox ────────────────────────────────────────────

    def on_decision(self, approval) -> None:
        task = self.store.get_task(approval.task_id) if approval.task_id else None
        try:
            if approval.kind == "kaggle_submit":
                self._handle_submit(approval, task)
                return
            if approval.kind == "send_email":
                self._handle_send(approval)
                return
            if task is None or approval.kind in ("done", "digest"):
                return
            resume = task.state.get("resume_phase", "work")
            if approval.kind == "review":
                if approval.status == "approved":
                    self._next(task, "finalize")
                else:
                    self.store.update_task(task.id, attempts=0)
                    self._next(task, "work", feedback=f"Dalton reviewed your result and sent it back:\n{approval.response or '(no note)'}")
                return
            # questions / escalations
            if approval.status == "approved":
                if approval.payload.get("switch_to"):
                    self.store.update_task(task.id, worker=approval.payload["switch_to"])
                self.store.update_task(task.id, attempts=0)
                fb = f"Dalton answered: {approval.response}" if approval.response else None
                self._next(task, resume, feedback=fb)
            else:
                self.store.log(task.id, "paused", f"You declined; task paused. {approval.response or ''}".strip())
                self._set(task, PAUSED)
        finally:
            self.store.mark_handled(approval.id)

    def _handle_send(self, approval) -> None:
        p = approval.payload
        if approval.status != "approved":
            self.store.log(approval.task_id, "not_sent",
                           f"Not sent: {p.get('subject')}. The draft stays in your Gmail Drafts folder.")
            return
        try:
            msg_id = self.gmail.send_draft(p["draft_id"])
            self.store.log(approval.task_id, "sent", f"Sent to {p.get('to')}: {p.get('subject')} (id {msg_id})")
        except Exception as e:
            self.store.log(approval.task_id, "send_error", f"Sending '{p.get('subject')}' failed: {e}")
            self.store.add_approval(approval.task_id, "done", f"Sending FAILED: {p.get('subject')} — {e}", {})

    def _handle_submit(self, approval, task: Task | None) -> None:
        p = approval.payload
        if approval.status != "approved":
            if task is not None:
                self.store.log(task.id, "submit_declined", f"Submission declined. {approval.response or ''}".strip())
            return
        from autobot.computer.kaggle_tool import Kaggle
        k = Kaggle()
        comp = p.get("competition")
        msg = p.get("message") or f"Autobot task #{approval.task_id}"
        try:
            if p.get("kernel") and p.get("version") is not None:
                out = k.submit_code_competition(comp, p["kernel"], int(p["version"]), p.get("file") or "submission.csv", msg)
            else:
                path = p.get("file")
                if path and not os.path.isabs(path) and p.get("cwd"):
                    path = os.path.join(p["cwd"], path)
                out = k.submit(comp, path, msg)
            self.store.log(approval.task_id, "submitted", f"Submitted to {comp}: {str(out)[:500]}")
            self.store.add_approval(approval.task_id, "done", f"Submitted to Kaggle '{comp}'", {"result": str(out)[:2000]})
        except Exception as e:
            self.store.log(approval.task_id, "submit_error", f"Submission failed: {e}")
            self.store.add_approval(approval.task_id, "done", f"Kaggle submission FAILED for '{comp}': {e}", {})

    # ── worker failure handling ─────────────────────────────────────────────

    def _cooling(self, task: Task, worker: str) -> bool:
        until = self.store.cooldown_until(worker)
        if until > time.time():
            self._set(task, WAITING, until - time.time())
            return True
        return False

    def _handle_worker_failure(self, task: Task, res: WorkerResult, retry_phase: str) -> bool:
        """Failures that are about the tool, not the task. True if handled."""
        if res.error_class == "rate_limited":
            until = time.time() + RATE_LIMIT_COOLDOWN_S
            self.store.set_cooldown(res.worker, until, res.error[:200])
            self._set(task, WAITING, RATE_LIMIT_COOLDOWN_S, phase=retry_phase)
            return True
        if res.error_class in ("auth", "not_installed", "unsupported"):
            self._ask(task, f"{res.worker} can't run: {res.error[:400]}\nFix it (e.g. sign in), then approve to retry.",
                      resume_phase=retry_phase)
            return True
        return False

    # ── briefs ──────────────────────────────────────────────────────────────

    def _brief(self, task: Task, lane: LaneSpec) -> str:
        crit = task.criteria
        if crit:
            checks_text = "HOW SUCCESS WILL BE CHECKED (Autobot runs these itself after you finish):\n" + \
                "\n".join(f"- {checks_mod.describe(c)}" for c in crit)
        elif lane.rounds:
            checks_text = "This task runs in rounds. Each round: do one solid step, push kernels if needed, report."
        else:
            checks_text = "No automatic checks exist for this task; Dalton will review your result."
        extras = []
        if task.config.get("style"):
            extras.append(f"STYLE (from Dalton):\n{task.config['style']}")
        if task.config.get("spec"):
            extras.append(f"SPECIFICATION:\n{task.config['spec']}")
        if task.config.get("outline"):
            extras.append("OUTLINE (each item must be a section heading):\n" + "\n".join(f"- {o}" for o in task.config["outline"]))
        if task.config.get("competition"):
            extras.append(f"COMPETITION: {task.config['competition']}")
        if lane.name == "email":
            new_mail = task.state.get("new_mail") or []
            if new_mail:
                extras.append("TRIAGE MODE. New messages to read (paths relative to the project folder):\n"
                              + "\n".join(f"- {m}" for m in new_mail)
                              + f"\nWrite the digest to: {task.state.get('digest_path')}")
            else:
                extras.append("COMPOSE MODE. Write the requested emails as draft files in drafts/.")
            style_file = None
            try:
                from autobot.paths import autobot_home
                style_file = autobot_home() / "email_style.md"
            except Exception:
                pass
            if style_file and style_file.exists():
                extras.append("DALTON'S EMAIL STYLE NOTES:\n" + style_file.read_text(encoding="utf-8")[:4000])
        return "\n\n".join(filter(None, [
            f"You are one of Dalton's tools. Autobot, his task manager, is handing you task #{task.id}.",
            f"TASK: {task.title}",
            f"WHAT DALTON WANTS (his words):\n{task.intent}",
            f"PROJECT FOLDER: {self._cwd(task)}",
            *extras,
            lane.guidance.replace("{KAGGLE_CLI}", _kaggle_cli_cmd()),
            checks_text,
            COMMON_RULES,
        ]))

    def _followup(self, task: Task, feedback: str) -> str:
        return (f"Continuing task #{task.id} ({task.title}).\n\n{feedback}\n\n"
                "When you're done, end with the structured report again.")

    # ── git helpers (coding lane) ───────────────────────────────────────────

    def _maybe_branch(self, task: Task, cwd: str) -> str | None:
        from autobot.integrations.cli_exec import run_cli
        if not os.path.isdir(os.path.join(cwd, ".git")):
            return None
        cur = run_cli(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, timeout=30)
        if not cur.ok:
            return None
        branch = f"autobot/task-{task.id}"
        if cur.stdout.strip() == branch:
            return f"on branch {branch}"
        dirty = run_cli(["git", "status", "--porcelain"], cwd=cwd, timeout=30)
        if dirty.ok and dirty.stdout.strip():
            return f"working tree has uncommitted changes, so staying on {cur.stdout.strip()}"
        base = run_cli(["git", "rev-parse", "HEAD"], cwd=cwd, timeout=30).stdout.strip()
        made = run_cli(["git", "checkout", "-b", branch], cwd=cwd, timeout=30)
        if made.ok:
            self.store.merge_state(task.id, base_commit=base, branch=branch, base_branch=cur.stdout.strip())
            return f"created branch {branch} from {cur.stdout.strip()}"
        return None

    def _diffstat(self, task: Task) -> str:
        from autobot.integrations.cli_exec import run_cli
        cwd = self._cwd(task)
        if not cwd or not os.path.isdir(os.path.join(cwd, ".git")):
            return ""
        out = []
        base = task.state.get("base_commit")
        if base:
            r = run_cli(["git", "diff", "--stat", f"{base}..HEAD"], cwd=cwd, timeout=30)
            if r.ok and r.stdout.strip():
                out.append(r.stdout.strip())
        r = run_cli(["git", "status", "--short"], cwd=cwd, timeout=30)
        if r.ok and r.stdout.strip():
            out.append("Uncommitted:\n" + r.stdout.strip())
        return "\n".join(out)[:3000]


def _kaggle_cli_cmd() -> str:
    from autobot.kaggle_cli import script_path
    return f'python "{script_path()}"'


# ── Kaggle status plumbing (kept out of the class so tests can inject fakes) ──

def _kaggle_statuses(kernels: list[str]) -> dict[str, dict]:
    from autobot.computer.kaggle_watchdog import KaggleJobLedger, make_kaggle_api, poll_pending
    ledger = KaggleJobLedger()
    try:
        poll_pending(make_kaggle_api(), ledger)
        ledger.load()
    except Exception as e:
        logger.warning(f"kaggle status poll failed: {e}")
    out = {}
    for k in kernels:
        j = ledger.get(k)
        out[k] = {"status": j.status, "error_log": j.error_log} if j else {"status": "unknown"}
    return out


def _ledger_kernels_since(started: float, cwd: str | None) -> list[str]:
    """Kernels registered in the ledger since `started` from inside this task's folder —
    so a kernel the worker pushed but forgot to report is still waited on."""
    try:
        from autobot.computer.kaggle_watchdog import KaggleJobLedger
        ledger = KaggleJobLedger()
    except Exception:
        return []
    root = str(Path(cwd).resolve()).lower() if cwd else None
    found = []
    for j in ledger.all():
        if j.dispatched_at >= started - 5:
            if root is None or (j.path and str(Path(j.path).resolve()).lower().startswith(root)):
                found.append(j.kernel)
    return found
