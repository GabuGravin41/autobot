"""
Agent Output Models — Structured data models for the agent's LLM responses.

Adapted from Browser Use's agent/views.py. These Pydantic models define
the expected structured output from the LLM at each step.

The LLM must return:
    - thinking: chain-of-thought reasoning
    - evaluation_previous_goal: did the last action work?
    - memory: key facts to remember
    - next_goal: what to do next
    - action: list of actions to execute
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Action Models
# ─────────────────────────────────────────────

class NavigateAction(BaseModel):
    """Navigate to a URL."""
    url: str


class ClickAction(BaseModel):
    """Click an element by its DOM index."""
    index: int


class InputTextAction(BaseModel):
    """Type text into an element by its DOM index."""
    index: int
    text: str


class ScrollAction(BaseModel):
    """Scroll the page."""
    amount: int = 3  # Number of scroll units


class PressKeyAction(BaseModel):
    """Press a keyboard key."""
    key: str  # e.g., "Enter", "Tab", "Escape"


class SwitchTabAction(BaseModel):
    """Switch to a browser tab."""
    tab_id: str


class NewTabAction(BaseModel):
    """Open a new tab."""
    url: str = "about:blank"


class WaitAction(BaseModel):
    """Wait for a specified duration."""
    seconds: float = 2.0


class DoneAction(BaseModel):
    """Complete the task."""
    text: str = ""  # Summary of results
    success: bool = True


class ScreenshotAction(BaseModel):
    """Take a screenshot for visual verification."""
    pass


class GoBackAction(BaseModel):
    """Go back one page."""
    pass


class CloseTabAction(BaseModel):
    """Close the current tab."""
    pass


# ─────────────────────────────────────────────
# Action Union — all possible actions the agent can take
# ─────────────────────────────────────────────

class ActionModel(BaseModel):
    """
    A single action the agent wants to execute.

    The LLM outputs one of these for each action in the step.
    Using Optional fields so the LLM outputs exactly one action type.

    Example LLM output:
        {"click": {"index": 4}}
        {"input_text": {"index": 2, "text": "hello"}}
        {"navigate": {"url": "https://google.com"}}
    """
    navigate: NavigateAction | None = None
    click: ClickAction | None = None
    input_text: InputTextAction | None = None
    scroll_down: ScrollAction | None = None
    scroll_up: ScrollAction | None = None
    press_key: PressKeyAction | None = None
    switch_tab: SwitchTabAction | None = None
    new_tab: NewTabAction | None = None
    close_tab: CloseTabAction | None = None
    wait: WaitAction | None = None
    done: DoneAction | None = None
    screenshot: ScreenshotAction | None = None
    go_back: GoBackAction | None = None

    @property
    def action_name(self) -> str:
        """Get the name of the active action."""
        for field_name in self.model_fields:
            if getattr(self, field_name) is not None:
                return field_name
        return "unknown"

    @property
    def action_data(self) -> BaseModel | None:
        """Get the active action's data."""
        for field_name in self.model_fields:
            val = getattr(self, field_name)
            if val is not None:
                return val
        return None

    @property
    def is_page_changing(self) -> bool:
        """
        Whether this action might change the page.
        If True, remaining actions in the step should be skipped after execution.
        (Adapted from Browser Use's action category system.)
        """
        return self.action_name in {
            "navigate", "go_back", "switch_tab", "new_tab", "close_tab",
        }


# ─────────────────────────────────────────────
# Agent Output — the complete structured response from the LLM
# ─────────────────────────────────────────────

class AgentOutput(BaseModel):
    """
    The complete structured output from the LLM at each step.

    Adapted from Browser Use's structured JSON output requirement.
    Forces the LLM to think, evaluate, remember, plan, then act.
    """
    thinking: str = Field(
        description="Step-by-step reasoning about current state and what to do."
    )
    evaluation_previous_goal: str = Field(
        default="",
        description="One sentence: did the last action succeed or fail?"
    )
    memory: str = Field(
        default="",
        description="1-3 sentences of key facts to remember for future steps."
    )
    next_goal: str = Field(
        description="One clear sentence: what you will do next and why."
    )
    action: list[ActionModel] = Field(
        description="List of actions to execute sequentially."
    )


# ─────────────────────────────────────────────
# Action Result — feedback after executing an action
# ─────────────────────────────────────────────

class ActionResult(BaseModel):
    """Result of executing a single action."""
    action_name: str
    success: bool
    error: str | None = None
    extracted_content: str | None = None
    page_changed: bool = False


# ─────────────────────────────────────────────
# Step Info — metadata about the current step
# ─────────────────────────────────────────────

class AgentStepInfo(BaseModel):
    """Metadata for the current step in the agent loop."""
    step_number: int
    max_steps: int
    goal: str


# ─────────────────────────────────────────────
# History Entry — record of what happened at each step
# ─────────────────────────────────────────────

class StepHistoryEntry(BaseModel):
    """
    Record of one step in the agent's history.
    Includes what the agent thought, what it did, and what happened.
    """
    step_number: int
    agent_output: AgentOutput
    action_results: list[ActionResult]
    url_before: str
    url_after: str

    def to_history_text(self) -> str:
        """
        Convert this step to a text summary for the agent history.
        Adapted from Browser Use's step history formatting.
        """
        lines = [f"Step {self.step_number + 1}:"]
        lines.append(f"  Goal: {self.agent_output.next_goal}")

        for i, (action, result) in enumerate(
            zip(self.agent_output.action, self.action_results)
        ):
            status = "✅" if result.success else "❌"
            action_desc = f"{action.action_name}"
            if action.action_data:
                # Get a brief description of the action parameters
                data = action.action_data
                if isinstance(data, ClickAction):
                    action_desc = f"click(index={data.index})"
                elif isinstance(data, InputTextAction):
                    action_desc = f"input_text(index={data.index}, text='{data.text[:30]}')"
                elif isinstance(data, NavigateAction):
                    action_desc = f"navigate(url='{data.url[:50]}')"
                elif isinstance(data, DoneAction):
                    action_desc = f"done(success={data.success})"

            error_text = f" Error: {result.error}" if result.error else ""
            lines.append(f"  Action {i + 1}: {status} {action_desc}{error_text}")

        if self.url_before != self.url_after:
            lines.append(f"  URL changed: {self.url_before[:50]} → {self.url_after[:50]}")

        return "\n".join(lines)
