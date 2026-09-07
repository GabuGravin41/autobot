"""
StopCondition — Describes when an agent run should halt.

JSON-serializable for checkpointing. Supports four modes:
  "steps"   — stop after N steps
  "metric"  — stop when a tracked metric crosses a threshold
  "time"    — stop after N wall-clock seconds
  "none"    — perpetual; never stop autonomously (Kaggle bots, monitors)
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class StopCondition(BaseModel):
    """
    Stopping criterion for an agent run.

    Pass one of these to AgentLoop. EvaluationAgent checks it every N steps.
    """

    type: Literal["steps", "metric", "time", "none"] = "steps"

    # ── steps mode ──────────────────────────────────────────────────────────
    max_steps: int | None = Field(
        default=None,
        description="Stop after this many steps (type='steps').",
    )

    # ── metric mode ─────────────────────────────────────────────────────────
    metric_key: str | None = Field(
        default=None,
        description="Key in the metrics context dict to watch (type='metric').",
    )
    metric_threshold: float | None = Field(
        default=None,
        description="Stop when context[metric_key] >= metric_threshold.",
    )
    metric_description: str | None = Field(
        default=None,
        description="Human-readable description, e.g. 'Make 5 Kaggle submissions'.",
    )

    # ── time mode ───────────────────────────────────────────────────────────
    max_seconds: float | None = Field(
        default=None,
        description="Stop after this many wall-clock seconds (type='time').",
    )
    _start_time: float | None = None  # set when the run starts

    # ── shared ──────────────────────────────────────────────────────────────
    description: str = Field(
        default="",
        description="Natural-language summary shown in the UI.",
    )

    def start_timer(self) -> None:
        """Call when the run begins (for time-based conditions)."""
        object.__setattr__(self, "_start_time", time.time())

    def is_met(self, context: dict[str, Any]) -> bool:
        """
        Return True if the stop condition has been satisfied.

        Args:
            context: dict with keys like:
                - "step_number": int
                - "metrics": dict[str, float]  (agent-tracked values)
                - "elapsed_seconds": float
        """
        if self.type == "none":
            return False

        if self.type == "steps":
            if self.max_steps is None:
                return False
            return context.get("step_number", 0) >= self.max_steps

        if self.type == "time":
            if self.max_seconds is None:
                return False
            elapsed = context.get("elapsed_seconds", 0.0)
            return elapsed >= self.max_seconds

        if self.type == "metric":
            if self.metric_key is None or self.metric_threshold is None:
                return False
            metrics = context.get("metrics", {})
            value = metrics.get(self.metric_key, 0.0)
            return float(value) >= self.metric_threshold

        return False

    def progress_text(self, context: dict[str, Any]) -> str:
        """Human-readable progress toward the stop condition."""
        if self.type == "none":
            step = context.get("step_number", 0)
            return f"Perpetual mode — step {step} (no limit)"

        if self.type == "steps":
            step = context.get("step_number", 0)
            total = self.max_steps or "?"
            pct = int(step / (self.max_steps or 1) * 100)
            return f"Step {step}/{total} ({pct}%)"

        if self.type == "time":
            elapsed = context.get("elapsed_seconds", 0.0)
            remaining = max(0.0, (self.max_seconds or 0) - elapsed)
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            return f"{mins}m {secs}s remaining"

        if self.type == "metric":
            metrics = context.get("metrics", {})
            value = metrics.get(self.metric_key or "", 0.0)
            threshold = self.metric_threshold or 0
            desc = self.metric_description or f"{self.metric_key} >= {threshold}"
            return f"{desc} — current: {value}/{threshold}"

        return "unknown"


# ── Convenience constructors ─────────────────────────────────────────────────

def perpetual(description: str = "Run until goal is achieved") -> StopCondition:
    """No step limit — run forever until agent calls done() or metric met."""
    return StopCondition(type="none", description=description)


def after_steps(n: int, description: str = "") -> StopCondition:
    """Stop after exactly N steps."""
    return StopCondition(
        type="steps",
        max_steps=n,
        description=description or f"Run for up to {n} steps",
    )


def after_seconds(seconds: float, description: str = "") -> StopCondition:
    """Stop after N wall-clock seconds."""
    mins = int(seconds // 60)
    return StopCondition(
        type="time",
        max_seconds=seconds,
        description=description or f"Run for up to {mins} minutes",
    )


def when_metric(
    key: str,
    threshold: float,
    description: str = "",
) -> StopCondition:
    """Stop when a tracked metric reaches a threshold."""
    return StopCondition(
        type="metric",
        metric_key=key,
        metric_threshold=threshold,
        metric_description=description or f"{key} >= {threshold}",
        description=description or f"Run until {key} reaches {threshold}",
    )
