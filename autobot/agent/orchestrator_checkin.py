"""
Orchestrator Check-In — the read-mostly half of the multi-project
orchestrator feature (Sep 2026): ask each tracked project's AI backend
(Claude Code or Antigravity, both headless — see
autobot/computer/claude_code_tool.py / antigravity_tool.py) for a status
update, and record it, so the user can ask Autobot "what's the state of my
seqoy project" instead of alt-tabbing into VS Code and re-reading the
conversation themselves.

Deliberately scoped to checking in, not directing. Sending an actual
follow-up instruction — "go implement X now" — on the user's behalf while
they're not watching is a materially bigger decision than reading a status
report: it can cause real file writes and real time/token spend in the
target project. The Kaggle side of this session's work got an explicit,
considered answer from the user for its unattended-autonomy question (see
approval.py's "Unattended mode" docstring); this feature hasn't, so
sending autonomous follow-up prompts stays out of this module until that
same kind of explicit decision exists for it. What's here — read the
status, summarize it, remember it — is useful on its own and carries none
of that risk, so it doesn't need to wait for the harder design conversation
to ship.

Why this module calls the bridge functions (claude_code_bridge.run_headless
/ antigravity_bridge.run_headless) directly instead of going through
computer.claude_code.run() / computer.antigravity.run(): those wrappers
exist for the LLM-facing computer_call path (see their own docstrings) and
deliberately collapse the bridge's full {"ok","data","error"} result down
to a plain string, because that's what dispatch.py needs to hand back to
the model. This module is plain orchestration Python, not an LLM tool
call, and it specifically needs the session_id / conversation_id field
the string-collapsed version throws away — calling the bridge directly is
not a shortcut around the tool layer, it's using the layer this code
actually belongs to.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from autobot.knowledge.project_registry import ProjectRegistry, TrackedProject

logger = logging.getLogger(__name__)

_CHECK_IN_PROMPT_TEMPLATE = """You are being asked for a brief status check-in, not new work.

Here is what the user originally told me they want from this project:
---
{intent_context}
---

Give a concise status update: what's been completed, what's currently in
progress, what's blocked (and on what), and anything that genuinely needs
a decision from the user before you can continue. Do not start new work
in response to this message — this is a status read, not an instruction."""


@dataclass
class CheckInResult:
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


class OrchestratorCheckIn:
    """Check in on one or every tracked project's AI backend."""

    def __init__(self, registry: ProjectRegistry | None = None) -> None:
        self.registry = registry or ProjectRegistry()

    async def check_in_on(self, project_name: str, timeout: float = 300.0) -> CheckInResult:
        """
        Ask one tracked project's AI backend for a status update.

        Resumes the project's prior conversation (via its stored
        session_id) when one exists, so the AI has context from earlier
        check-ins rather than treating every check-in as a cold start.
        On success, records the summary and any new session_id back to
        the registry — so the NEXT check-in resumes THIS one, keeping the
        chain unbroken run over run.
        """
        project = self.registry.get(project_name)
        if project is None:
            return CheckInResult(
                project_name=project_name,
                backend="",
                ok=False,
                summary="",
                error=f"No tracked project named {project_name!r}. "
                      f"Registered: {[p.name for p in self.registry.list_all()]}",
            )

        prompt = _CHECK_IN_PROMPT_TEMPLATE.format(intent_context=project.full_intent_context())

        try:
            if project.backend == "claude_code":
                result = await self._check_in_claude_code(project, prompt, timeout)
            elif project.backend == "antigravity":
                result = await self._check_in_antigravity(project, prompt, timeout)
            else:
                # Should be unreachable — ProjectRegistry.register() already
                # validates backend against KNOWN_BACKENDS — but a registry
                # file edited by hand or from an older schema version could
                # still produce one. Fail with a clear message, not a
                # dispatch-by-string KeyError three frames down.
                return CheckInResult(
                    project_name=project.name, backend=project.backend, ok=False, summary="",
                    error=f"Unrecognized backend {project.backend!r} on stored project — "
                          f"registry data may be from an incompatible version.",
                )
        except Exception as e:
            logger.warning(f"Check-in on {project.name!r} raised: {e}")
            return CheckInResult(project_name=project.name, backend=project.backend, ok=False, summary="", error=str(e))

        if result.ok:
            self.registry.record_check_in(project.name, result.summary)
        return result

    async def check_in_on_all(self, timeout: float = 300.0) -> list[CheckInResult]:
        """
        Check in on every tracked project. One project's failure (backend
        not installed, timed out, whatever) does not stop the others from
        being checked — the whole point of this being a batch operation is
        that the user gets a full picture even if one project's AI backend
        is temporarily unreachable, not silence because the third project
        in the list happened to error.
        """
        results: list[CheckInResult] = []
        for project in self.registry.list_all():
            results.append(await self.check_in_on(project.name, timeout=timeout))
        return results

    # ── per-backend dispatch ─────────────────────────────────────────────

    async def _check_in_claude_code(self, project: TrackedProject, prompt: str, timeout: float) -> CheckInResult:
        import asyncio
        from autobot.integrations import claude_code_bridge

        # permission_mode="plan" (the bridge's own default) is exactly
        # right for a check-in: read-only, so even if the model interprets
        # "give a status update" loosely, headless mode structurally can't
        # write anything without an interactive prompt it can't answer.
        result = await asyncio.to_thread(
            claude_code_bridge.run_headless,
            prompt=prompt,
            cwd=project.working_dir,
            resume_session_id=project.session_id,
            timeout=timeout,
        )
        return self._result_from_bridge(project, result, text_key="result", id_key="session_id")

    async def _check_in_antigravity(self, project: TrackedProject, prompt: str, timeout: float) -> CheckInResult:
        import asyncio
        from autobot.integrations import antigravity_bridge

        result = await asyncio.to_thread(
            antigravity_bridge.run_headless,
            prompt=prompt,
            cwd=project.working_dir,
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
    ) -> CheckInResult:
        if not bridge_result.get("ok"):
            return CheckInResult(
                project_name=project.name, backend=project.backend, ok=False, summary="",
                error=bridge_result.get("error", "check-in failed"),
            )

        data = bridge_result.get("data")
        if isinstance(data, dict):
            summary = str(data.get(text_key, data))
            new_session_id = data.get(id_key)
            if new_session_id and new_session_id != project.session_id:
                self.registry.update_session(project.name, new_session_id)
        else:
            summary = str(data)

        return CheckInResult(project_name=project.name, backend=project.backend, ok=True, summary=summary)
