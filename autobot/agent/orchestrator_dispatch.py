"""
Orchestrator Dispatch — the "run multiple projects at once" half of the
multi-project orchestrator, deliberately built as a separate module from
orchestrator_checkin.py rather than an extension of it.

orchestrator_checkin.py's own module docstring drew this line on purpose
back in Round 6: reading a status update is a fundamentally smaller
decision than sending a real follow-up instruction — the latter can cause
real file writes and real time/token spend in a project the user isn't
watching, across as many projects as are tracked, all at once. That
docstring said explicitly: "this feature hasn't [gotten an explicit,
considered decision, unlike Kaggle's unattended-autonomy question], so
sending autonomous follow-up prompts stays out of this module until that
same kind of explicit decision exists for it." Dalton asking directly for
"ability to run multiple projects like openclaw" (Sep 2026) IS that
decision. This module is the result — kept separate from
orchestrator_checkin.py rather than merged into it, so the read-only
check-in path (already shipped, already safe by construction) is never
put at risk by a change made in service of the write-capable path.

The OpenClaw reference (from s6e9_ev_prediction/THINKING_AND_DECISIONS.md
section 5) is a pattern, not a library — nothing here talks to OpenClaw.
Three specific ideas were worth actually taking:

1. **Stateless turns, stateful workspace.** Each project's dispatch is one
   bounded subprocess call (via claude_code_bridge/antigravity_bridge,
   same as check-in), not a persistent thread — the ledger-on-disk pattern
   from kaggle_watchdog.py already establishes this project's take on
   "state lives on disk, not in a process," and this module follows it:
   ProjectRegistry.update_session()/record_check_in() are what let the
   NEXT dispatch resume the right conversation, not any in-memory state
   this module holds itself.
2. **Strict lane isolation.** One project's dispatch failing (backend
   crashed, cwd got deleted, timed out) must never affect another
   project's — see dispatch_on_all()'s per-task try/except, mirroring
   check_in_on_all()'s existing isolation.
3. **Global concurrency throttling.** OpenClaw's Gateway caps how many
   LLM executions run at once and queues the rest, rather than firing off
   unbounded parallel work. dispatch_on_all() does the same with a plain
   asyncio.Semaphore (AUTOBOT_ORCHESTRATOR_MAX_CONCURRENT, default 3) —
   deliberately NOT the fuller "hardware slot" semaphore
   kaggle_watchdog.py's check_capacity() implements for Kaggle GPU/CPU
   kernels specifically. That one enforces a real, hard, externally-owned
   resource limit (Kaggle's account-wide kernel caps) and lives in
   kaggle_tool.py where the actual dispatch happens — a Kaggle-heavy
   project's push_kernel() call is already protected by it regardless of
   how many projects this module runs concurrently. This module's own
   semaphore is a softer, local one: how many headless CLI subprocesses
   (claude/agy) this machine should run at once, which is a CPU/RAM
   question, not a Kaggle-account question, and applies to every backend,
   not just Kaggle-flavored projects.

Safety default, inherited rather than reinvented: this module passes
whatever permission_mode / skip_permissions the CALLER explicitly supplies
straight through to claude_code_bridge.run_headless /
antigravity_bridge.run_headless, whose own defaults are already the safe
ones ("plan" / skip_permissions=False — read-only). It does not choose a
more permissive default on projects' behalf. A caller (a human running
`autobot --dispatch-all`, or the dashboard) that wants write access on a
particular project asks for it explicitly, per project, same as any other
DANGER-tier decision elsewhere in this codebase.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from autobot.knowledge.project_registry import ProjectRegistry, TrackedProject

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = int(os.getenv("AUTOBOT_ORCHESTRATOR_MAX_CONCURRENT", "3"))


@dataclass
class DispatchResult:
    project_name: str
    backend: str
    ok: bool
    summary: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "backend": self.backend,
            "ok": self.ok,
            "summary": self.summary,
            "error": self.error,
        }


class OrchestratorDispatch:
    """Send a real instruction to one or many tracked projects' AI
    backends, running multiple projects concurrently when asked to."""

    def __init__(self, registry: ProjectRegistry | None = None) -> None:
        self.registry = registry or ProjectRegistry()

    async def dispatch_on(
        self,
        project_name: str,
        instruction: str,
        timeout: float = 600.0,
        permission_mode: str = "plan",
        skip_permissions: bool = False,
    ) -> DispatchResult:
        """
        Send ONE real instruction to one tracked project's AI backend and
        record the result. Resumes the project's prior session when one
        exists, exactly like check_in_on() — a dispatched instruction and
        a check-in are both just "a turn in this project's conversation"
        from the backend's point of view, they differ only in what the
        prompt asks for and what permission_mode allows it to do.

        permission_mode ("plan" default) is Claude Code's write-gating
        knob; skip_permissions (False default) is Antigravity's. Only the
        one matching the project's actual backend is used — passing both
        is harmless, the unused one is ignored by the branch that doesn't
        apply.
        """
        if not instruction or not instruction.strip():
            return DispatchResult(
                project_name=project_name, backend="", ok=False, summary="",
                error="dispatch_on: instruction is required",
            )

        project = self.registry.get(project_name)
        if project is None:
            return DispatchResult(
                project_name=project_name, backend="", ok=False, summary="",
                error=f"No tracked project named {project_name!r}. "
                      f"Registered: {[p.name for p in self.registry.list_all()]}",
            )

        try:
            if project.backend == "claude_code":
                result = await self._dispatch_claude_code(project, instruction, timeout, permission_mode)
            elif project.backend == "antigravity":
                result = await self._dispatch_antigravity(project, instruction, timeout, skip_permissions)
            else:
                return DispatchResult(
                    project_name=project.name, backend=project.backend, ok=False, summary="",
                    error=f"Unrecognized backend {project.backend!r} on stored project — "
                          f"registry data may be from an incompatible version.",
                )
        except Exception as e:
            logger.warning(f"Dispatch on {project.name!r} raised: {e}")
            return DispatchResult(project_name=project.name, backend=project.backend, ok=False, summary="", error=str(e))

        if result.ok:
            self.registry.record_check_in(project.name, result.summary)
        return result

    async def dispatch_on_all(
        self,
        instructions: dict[str, str],
        timeout: float = 600.0,
        max_concurrent: int | None = None,
        permission_modes: dict[str, str] | None = None,
        skip_permissions: dict[str, bool] | None = None,
    ) -> list[DispatchResult]:
        """
        Run multiple tracked projects CONCURRENTLY — the actual "run
        multiple projects like OpenClaw" capability. `instructions` maps
        project name -> the real instruction to send it; only projects
        named here are touched, so this is opt-in per call, never "dispatch
        work to everything I've ever tracked."

        Bounded by an asyncio.Semaphore (max_concurrent, default
        AUTOBOT_ORCHESTRATOR_MAX_CONCURRENT / 3) — OpenClaw's "Global
        Concurrency Throttling" pattern: excess dispatches queue rather
        than firing every headless CLI subprocess at once, which on a real
        machine competes for the same CPU/RAM regardless of which project
        each one belongs to.

        One project's failure never stops or corrupts another's result —
        each dispatch is independently try/excepted (inside dispatch_on())
        AND independently awaited via asyncio.gather(return_exceptions=True)
        as a second layer, so even a bug in this method's own bookkeeping
        for one project can't take the whole batch down.

        permission_modes / skip_permissions are optional per-project
        overrides (keyed the same as `instructions`) for callers that need
        write access on SOME projects and not others in the same batch —
        omitted projects keep the safe defaults ("plan" / False).
        """
        permission_modes = permission_modes or {}
        skip_permissions = skip_permissions or {}
        semaphore = asyncio.Semaphore(max_concurrent or DEFAULT_MAX_CONCURRENT)

        async def _bounded_dispatch(name: str, instruction: str) -> DispatchResult:
            async with semaphore:
                return await self.dispatch_on(
                    name, instruction, timeout=timeout,
                    permission_mode=permission_modes.get(name, "plan"),
                    skip_permissions=skip_permissions.get(name, False),
                )

        tasks = [_bounded_dispatch(name, instr) for name, instr in instructions.items()]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[DispatchResult] = []
        for name, r in zip(instructions.keys(), raw_results):
            if isinstance(r, Exception):
                logger.warning(f"dispatch_on_all: {name!r} raised outside dispatch_on's own try/except: {r}")
                results.append(DispatchResult(project_name=name, backend="", ok=False, summary="", error=str(r)))
            else:
                results.append(r)
        return results

    # ── per-backend dispatch ─────────────────────────────────────────────

    async def _dispatch_claude_code(
        self, project: TrackedProject, instruction: str, timeout: float, permission_mode: str,
    ) -> DispatchResult:
        from autobot.integrations import claude_code_bridge

        result = await asyncio.to_thread(
            claude_code_bridge.run_headless,
            prompt=instruction,
            cwd=project.working_dir,
            permission_mode=permission_mode,
            resume_session_id=project.session_id,
            timeout=timeout,
        )
        return self._result_from_bridge(project, result, text_key="result", id_key="session_id")

    async def _dispatch_antigravity(
        self, project: TrackedProject, instruction: str, timeout: float, skip_permissions: bool,
    ) -> DispatchResult:
        from autobot.integrations import antigravity_bridge

        result = await asyncio.to_thread(
            antigravity_bridge.run_headless,
            prompt=instruction,
            cwd=project.working_dir,
            skip_permissions=skip_permissions,
            conversation_id=project.session_id,
            timeout=timeout,
        )
        return self._result_from_bridge(project, result, text_key="response", id_key="conversation_id")

    def _result_from_bridge(
        self,
        project: TrackedProject,
        bridge_result: dict[str, Any],
        text_key: str,
        id_key: str,
    ) -> DispatchResult:
        if not bridge_result.get("ok"):
            return DispatchResult(
                project_name=project.name, backend=project.backend, ok=False, summary="",
                error=bridge_result.get("error", "dispatch failed"),
            )

        data = bridge_result.get("data")
        if isinstance(data, dict):
            summary = str(data.get(text_key, data))
            new_session_id = data.get(id_key)
            if new_session_id and new_session_id != project.session_id:
                self.registry.update_session(project.name, new_session_id)
        else:
            summary = str(data)

        return DispatchResult(project_name=project.name, backend=project.backend, ok=True, summary=summary)
