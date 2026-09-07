"""
Action Models — Clean, flat data structures for the new CoreLoop.

Replaces the sprawling models.py with minimal dataclasses that are easy to
read, test, and understand. No Pydantic chains — just plain Python.

These models are the only shared data contract between CoreLoop, the runner,
the skill distiller, and the approval guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Action ────────────────────────────────────────────────────────────────────

@dataclass
class Action:
    """
    A single action the agent wants to take.

    name   — one of the action names listed in system_prompt.md
    params — keyword arguments for that action (always a dict)
    """
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Human-readable one-liner for logs."""
        if not self.params:
            return self.name
        parts = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.name}({parts})"


# ── Step record ───────────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    """
    Outcome of a single executed action.

    Used by SkillDistiller.distill_from_run(), which reads:
        entry.action_results  → list[ActionResult]
        result.success        → bool
        result.action_name    → str
        result.error          → str
    """
    action_name: str
    success: bool
    result: str = ""
    error: str = ""


@dataclass
class _AgentOutputStub:
    """Minimal stub so SkillDistiller.distill_from_run() can read entry.agent_output.next_goal."""
    next_goal: str


@dataclass
class StepRecord:
    """
    A complete record of one step in the agent loop.

    Compatible with SkillDistiller.distill_from_run(), which expects:
        entry.action_results       → list[ActionResult]
        entry.agent_output.next_goal → str
        entry.url_after            → str
    """
    step: int
    next_goal: str
    action: Action
    result: str
    success: bool
    screen_before: str
    screen_after: str
    url_after: str = ""          # empty for non-browser tasks

    # ─── SkillDistiller compatibility ─────────────────────────────────────────

    @property
    def action_results(self) -> list[ActionResult]:
        return [ActionResult(
            action_name=self.action.name,
            success=self.success,
            result=self.result,
            error="" if self.success else self.result,
        )]

    @property
    def agent_output(self) -> _AgentOutputStub:
        return _AgentOutputStub(next_goal=self.next_goal)

    def to_history_text(self) -> str:
        """One-line summary for dashboard status display."""
        icon = "✅" if self.success else "❌"
        return f"Step {self.step}: {icon} {self.action.describe()} — {self.next_goal}"


# ── LLM response ──────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Parsed output from one LLM call.

    The model always returns JSON with these three fields (see system_prompt.md).
    """
    thinking: str
    next_goal: str
    action: Action

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMResponse":
        raw_action = data.get("action", {})
        if isinstance(raw_action, dict):
            name = raw_action.get("name", "done")
            params = raw_action.get("params", {})
        else:
            name = "done"
            params = {"text": "Could not parse action", "success": False}
        return cls(
            thinking=str(data.get("thinking", "")),
            next_goal=str(data.get("next_goal", "")),
            action=Action(name=name, params=params),
        )
