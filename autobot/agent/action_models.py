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

# Every action CoreLoop can execute, with its positional parameter order.
# Used to validate action names and to map positional/list/scalar arguments
# from weaker models onto real parameter names (see autobot/util/jsonx.py).
ACTION_PARAMS: dict[str, list[str]] = {
    "click": ["index"],
    "type_into": ["index", "text"],
    "type": ["text"],
    "key": ["combo"],
    "focus": ["title"],
    "navigate": ["url"],
    "run_shell": ["command", "timeout"],
    "screenshot": [],
    "wait": ["seconds"],
    "human_input": ["prompt"],
    "computer_call": ["call"],
    "browser_text": [],
    "browser_list": [],
    "browser_click": ["index"],
    "browser_type": ["index", "text"],
    "browser_paste": ["index", "text"],
    "done": ["text", "success"],
}

_META_KEYS = ("thinking", "next_goal", "reasoning", "thought", "plan")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in ("true", "yes", "y", "1", "success", "succeeded")


@dataclass
class LLMResponse:
    """
    Parsed output from one LLM call.

    The model is asked for JSON with three fields (see system_prompt.md):
    thinking, next_goal, action{name, params}. from_dict() accepts that and
    the many near-miss shapes weaker models produce instead.

    It returns None for anything it can't map to a REAL action. It used to
    turn any unrecognized action shape into `done` (with success defaulting
    to True), which ended the run on step 1 and reported "Task complete" —
    reproduced with {"type": "click", "index": 1}. None makes CoreLoop re-ask
    instead.
    """
    thinking: str
    next_goal: str
    action: Action

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMResponse | None":
        from autobot.util.jsonx import coerce_call

        if not isinstance(data, dict):
            return None
        call = None
        raw_action = data.get("action")
        if raw_action is not None:
            call = coerce_call(raw_action, ACTION_PARAMS)
        if call is None:
            # Action fields placed at the top level next to thinking/next_goal.
            flat = {k: v for k, v in data.items() if k not in _META_KEYS}
            call = coerce_call(flat, ACTION_PARAMS)
        if call is None and isinstance(raw_action, dict) and isinstance(raw_action.get("name"), str) \
                and isinstance(raw_action.get("params", {}), dict) and raw_action["name"].strip():
            # Canonical shape with an unknown name: keep it, so the dispatcher
            # answers "Unknown action 'x'. Valid actions: ..." in the history.
            call = (raw_action["name"].strip(), dict(raw_action.get("params") or {}))
        if call is None:
            return None

        name, params = call
        if name == "done":
            # "done" only counts as success when the model says so explicitly.
            params = dict(params)
            params["success"] = _truthy(params.get("success", False))
            params.setdefault("text", "")

        thinking = data.get("thinking") or data.get("reasoning") or data.get("thought") or ""
        return cls(
            thinking=str(thinking),
            next_goal=str(data.get("next_goal", "") or data.get("plan", "") or ""),
            action=Action(name=name, params=params),
        )

    @classmethod
    def parse(cls, raw: str | None) -> "LLMResponse | None":
        """Parse raw model text (fenced, prose-wrapped, single-quoted...)."""
        from autobot.util.jsonx import coerce_call, extract_json

        data = extract_json(raw)
        if isinstance(data, dict):
            return cls.from_dict(data)
        if isinstance(raw, str):
            call = coerce_call(raw.strip().splitlines()[-1] if raw.strip() else "", ACTION_PARAMS)
            if call:
                return cls.from_dict({"action": {"name": call[0], "params": call[1]}})
        return None
