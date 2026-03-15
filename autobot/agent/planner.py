"""
ComplexityEstimator — Analyzes a task goal and returns a smart step budget.

Before spinning up an AgentLoop, the runner calls this to decide:
  - How many steps to allocate (or none for perpetual tasks)
  - What mode the task is (quick / research / perpetual)
  - What the stopping criterion should be

This replaces the fixed max_steps=100 default with a goal-aware budget.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from autobot.agent.stop_condition import (
    StopCondition,
    after_steps,
    after_seconds,
    perpetual,
    when_metric,
)

logger = logging.getLogger(__name__)


class ComplexityEstimate(BaseModel):
    """Output of ComplexityEstimator."""
    mode: str                         # "quick" | "research" | "perpetual"
    step_budget: int | None           # None = no step limit (perpetual)
    stop_condition: StopCondition
    reasoning: str                    # why this budget was assigned


# ── Heuristics used as fallback when LLM is unavailable ─────────────────────

_PERPETUAL_KEYWORDS = {
    "competition", "compete", "kaggle", "zindi", "monitor", "watch",
    "until i say stop", "keep running", "indefinitely", "forever",
    "continuously", "ongoing", "daily", "weekly", "schedule",
    "loop", "poll", "repeat", "submit multiple", "leaderboard",
}

_RESEARCH_KEYWORDS = {
    "research", "write", "paper", "report", "analyse", "analyze",
    "study", "investigate", "deep dive", "comprehensive", "multiple",
    "several", "gene annotation", "annotate", "blast", "ncbi",
    "literature review", "benchmark", "compare", "evaluate",
}

_QUICK_KEYWORDS = {
    "open", "find", "search", "check", "look", "navigate",
    "click", "go to", "read", "show", "get", "fetch",
    "send email", "type", "copy", "paste",
}

_METRIC_PATTERNS = [
    # "make 5 submissions" / "submit 5 times"
    (r"(\d+)\s+submissions?", "submissions", None),
    # "solve 20 problems"
    (r"(\d+)\s+problems?", "problems_solved", None),
    # "write 5 papers" / "5 research papers"
    (r"(\d+)\s+(?:research\s+)?papers?", "papers_written", None),
    # "top 10%" / "rank < 10" / "reach rank 10"
    (r"top\s+(\d+)%", "leaderboard_percentile", None),
    # "reach position N on leaderboard"
    (r"(?:rank|position)\s*[<≤]\s*(\d+)", "leaderboard_rank", None),
    # "50 websites"
    (r"(\d+)\s+websites?", "websites_built", None),
]


def _heuristic_estimate(goal: str) -> ComplexityEstimate:
    """Fast, LLM-free estimate based on keyword matching."""
    goal_lower = goal.lower()

    # Check for perpetual patterns first
    if any(kw in goal_lower for kw in _PERPETUAL_KEYWORDS):
        # Try to detect a metric stop condition
        for pattern, metric_key, _ in _METRIC_PATTERNS:
            m = re.search(pattern, goal_lower)
            if m:
                threshold = float(m.group(1))
                sc = when_metric(
                    key=metric_key,
                    threshold=threshold,
                    description=f"Reach {int(threshold)} {metric_key.replace('_', ' ')}",
                )
                return ComplexityEstimate(
                    mode="perpetual",
                    step_budget=None,
                    stop_condition=sc,
                    reasoning=f"Perpetual task with metric: {sc.description}",
                )
        # Perpetual with no detected metric
        return ComplexityEstimate(
            mode="perpetual",
            step_budget=None,
            stop_condition=perpetual("Run until goal is achieved or user stops"),
            reasoning="Perpetual task — no step limit",
        )

    # Research / multi-step tasks
    if any(kw in goal_lower for kw in _RESEARCH_KEYWORDS):
        return ComplexityEstimate(
            mode="research",
            step_budget=80,
            stop_condition=after_steps(80, "Research task — up to 80 steps"),
            reasoning="Research task detected — allocating 80 steps",
        )

    # Quick tasks
    return ComplexityEstimate(
        mode="quick",
        step_budget=25,
        stop_condition=after_steps(25, "Quick task — up to 25 steps"),
        reasoning="Quick task detected — allocating 25 steps",
    )


class ComplexityEstimator:
    """
    Uses the LLM to analyze a goal and return an appropriate step budget
    and stopping criterion.

    Falls back to keyword heuristics if the LLM is unavailable.
    """

    def __init__(self, llm_client: Any, model: str):
        self.llm_client = llm_client
        self.model = model

    async def estimate(self, goal: str) -> ComplexityEstimate:
        """Estimate the complexity of a task and return a step budget."""
        try:
            return await self._llm_estimate(goal)
        except Exception as e:
            logger.warning(f"ComplexityEstimator LLM failed: {e} — using heuristic fallback")
            return _heuristic_estimate(goal)

    async def _llm_estimate(self, goal: str) -> ComplexityEstimate:
        prompt = f"""You are a Task Complexity Analyzer for an autonomous desktop agent called Autobot.

Autobot can control ANY application on the computer — browser, VS Code, terminal, file manager, spreadsheets, etc.
It works by taking screenshots and using mouse/keyboard to interact with whatever is on screen.

## Task to Analyze
{goal}

## Your Job
Decide the right execution mode and stopping criterion.

### Modes
- **quick**: Simple, bounded task. Open an app, send a message, search for something, copy/paste. 10–30 steps.
- **research**: Multi-step investigation, writing, or analysis. Could involve multiple sites/apps. 40–100 steps.
- **perpetual**: Ongoing, open-ended, or competition-style task. Run until a specific metric is reached or the user stops it. No step limit.

### Stop Condition Types
- **steps**: "Run for up to N steps" — use for quick/research tasks
- **metric**: "Run until X reaches Y" — use for competitive tasks (5 Kaggle submissions, top 20% leaderboard)
- **time**: "Run for N hours" — use for monitoring/scheduled tasks
- **none**: "Run until done" — use for perpetual tasks with no clear metric

Respond with ONLY valid JSON:
{{
  "mode": "quick" | "research" | "perpetual",
  "step_budget": <integer or null if perpetual>,
  "stop_condition": {{
    "type": "steps" | "metric" | "time" | "none",
    "max_steps": <int or null>,
    "metric_key": "<key or null>",
    "metric_threshold": <number or null>,
    "metric_description": "<human description or null>",
    "max_seconds": <seconds or null>,
    "description": "<one sentence>"
  }},
  "reasoning": "<1-2 sentences explaining why>"
}}

Examples:
- "Search Google for machine learning papers" → quick, 20 steps, type=steps
- "Write a comprehensive report on climate change" → research, 80 steps, type=steps
- "Compete in the Titanic Kaggle competition until I reach top 20%" → perpetual, null steps, type=metric, metric_key=leaderboard_percentile, threshold=20
- "Monitor my Gmail and send me a daily summary" → perpetual, null steps, type=none
- "Make 5 submissions to the Kaggle Housing Prices competition" → perpetual, null steps, type=metric, metric_key=submissions, threshold=5"""

        resp = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""
        return self._parse(raw, goal)

    def _parse(self, raw: str, goal: str) -> ComplexityEstimate:
        """Parse LLM output into ComplexityEstimate."""
        text = raw.strip()

        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        start = text.find("{")
        if start != -1:
            end = text.rfind("}")
            text = text[start : end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("ComplexityEstimator: could not parse LLM JSON — using heuristic")
            return _heuristic_estimate(goal)

        # Parse stop condition
        sc_data = data.get("stop_condition", {})
        sc_type = sc_data.get("type", "steps")

        if sc_type == "metric":
            sc = StopCondition(
                type="metric",
                metric_key=sc_data.get("metric_key"),
                metric_threshold=sc_data.get("metric_threshold"),
                metric_description=sc_data.get("metric_description"),
                description=sc_data.get("description", ""),
            )
        elif sc_type == "time":
            sc = StopCondition(
                type="time",
                max_seconds=sc_data.get("max_seconds"),
                description=sc_data.get("description", ""),
            )
        elif sc_type == "none":
            sc = StopCondition(
                type="none",
                description=sc_data.get("description", "Perpetual task"),
            )
        else:  # steps
            budget = data.get("step_budget") or sc_data.get("max_steps") or 50
            sc = StopCondition(
                type="steps",
                max_steps=int(budget),
                description=sc_data.get("description", f"Up to {budget} steps"),
            )

        step_budget = data.get("step_budget")
        if step_budget is not None:
            step_budget = int(step_budget)

        return ComplexityEstimate(
            mode=data.get("mode", "quick"),
            step_budget=step_budget,
            stop_condition=sc,
            reasoning=data.get("reasoning", "LLM-estimated complexity"),
        )
