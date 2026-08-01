"""
EvaluationAgent — Mid-task progress monitor.

Runs every N steps during an AgentLoop to:
  - Detect when the agent is stuck or looping
  - Check whether the stop condition metric has been achieved
  - Signal: CONTINUE | REPLAN | COMPLETE | PAUSE | ESCALATE

Unlike JudgeAgent (runs once at the end), EvaluationAgent runs DURING
execution and can steer the agent back on course before problems compound.
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel

from autobot.agent.stop_condition import StopCondition

logger = logging.getLogger(__name__)


class EvalSignal(str, Enum):
    CONTINUE = "continue"    # Making progress — keep going
    REPLAN = "replan"        # Stuck — generate a fresh approach
    COMPLETE = "complete"    # Stop condition met — wrap up
    PAUSE = "pause"          # Needs human input before continuing
    ESCALATE = "escalate"    # Serious problem — alert user


class EvaluationResult(BaseModel):
    signal: EvalSignal
    reasoning: str
    new_plan: str | None = None          # Filled in when signal == REPLAN
    failed_because: str | None = None    # Root cause when signal == REPLAN
    alternatives: list[str] | None = None  # Ranked strategies when signal == REPLAN
    completion_summary: str | None = None  # Filled in when signal == COMPLETE
    alert_message: str | None = None     # Filled in when signal == ESCALATE/PAUSE


class EvaluationAgent:
    """
    Mid-task evaluator. Call `evaluate()` every N steps from AgentLoop.

    It reads the agent's recent history + scratchpad and decides whether
    the run should continue, replan, or stop.
    """

    def __init__(self, llm_client: Any, model: str):
        self.llm_client = llm_client
        self.model = model

    async def evaluate(
        self,
        goal: str,
        stop_condition: StopCondition,
        history_entries: list[Any],   # list[StepHistoryEntry]
        scratchpad: list[str],
        step_number: int,
        consecutive_failures: int,
        metrics: dict[str, float] | None = None,
        elapsed_seconds: float = 0.0,
    ) -> EvaluationResult:
        """
        Evaluate the current state of the agent run.

        Fast heuristic pre-check runs first — for obvious cases it avoids
        an expensive LLM call. The LLM is only called for ambiguous situations.

        Returns an EvaluationResult with a signal and reasoning.
        Falls back to CONTINUE if the LLM call fails.
        """
        sc_context = {
            "step_number": step_number,
            "metrics": metrics or {},
            "elapsed_seconds": elapsed_seconds,
        }
        stop_met = stop_condition.is_met(sc_context)

        # ── Fast heuristic pre-check (avoids expensive LLM call) ─────────────
        fast = self._fast_heuristic_check(
            step_number=step_number,
            consecutive_failures=consecutive_failures,
            stop_met=stop_met,
            history_entries=history_entries,
            scratchpad=scratchpad,
        )
        if fast is not None:
            logger.info(f"📊 EvaluationAgent [heuristic]: {fast.signal.value.upper()} — {fast.reasoning}")
            return fast

        # ── Full LLM evaluation ───────────────────────────────────────────────
        stop_progress = stop_condition.progress_text(sc_context)
        recent = self._summarize_history(history_entries[-15:])

        prompt = self._build_prompt(
            goal=goal,
            stop_condition=stop_condition,
            stop_progress=stop_progress,
            stop_met=stop_met,
            recent_history=recent,
            scratchpad=scratchpad,
            step_number=step_number,
            consecutive_failures=consecutive_failures,
            metrics=metrics or {},
            elapsed_seconds=elapsed_seconds,
        )

        try:
            result = await self._call_llm(prompt)
            return result
        except Exception as e:
            logger.warning(f"EvaluationAgent LLM call failed: {e} — defaulting to CONTINUE")
            return EvaluationResult(
                signal=EvalSignal.CONTINUE,
                reasoning=f"Evaluation skipped due to error: {e}",
            )

    def _fast_heuristic_check(
        self,
        step_number: int,
        consecutive_failures: int,
        stop_met: bool,
        history_entries: list[Any],
        scratchpad: list[str],
    ) -> EvaluationResult | None:
        """
        Heuristic evaluation — avoids LLM call for obvious decisions.

        Returns an EvaluationResult if the decision is clear, or None to
        proceed to full LLM evaluation.
        """
        # If stop condition is clearly met → COMPLETE
        if stop_met:
            return EvaluationResult(
                signal=EvalSignal.COMPLETE,
                reasoning="Stop condition metrics are satisfied.",
                completion_summary=f"Goal achieved at step {step_number}.",
            )

        # If agent is catastrophically stuck → REPLAN immediately
        if consecutive_failures >= 6:
            return EvaluationResult(
                signal=EvalSignal.REPLAN,
                reasoning=f"Agent has failed {consecutive_failures} consecutive times — current approach is not working.",
                new_plan="Stop the current approach entirely. Reassess from scratch.",
                failed_because=f"Same action/goal failed {consecutive_failures} consecutive times without progress.",
                alternatives=[
                    "Reassess the goal from scratch — what is the minimal path to completion?",
                    "Try a completely different tool or URL to reach the same destination",
                    "Skip this sub-task and continue with the next part of the goal",
                ],
            )

        # If very early in the run with no failures → CONTINUE (agent still exploring)
        if step_number < 8 and consecutive_failures == 0:
            return EvaluationResult(
                signal=EvalSignal.CONTINUE,
                reasoning=f"Early stage (step {step_number}), no failures — still making progress.",
            )

        # Check for goal loop in last 5 history entries
        if len(history_entries) >= 5:
            recent_goals = [
                e.agent_output.next_goal.strip().lower()[:60]
                for e in history_entries[-5:]
            ]
            if len(set(recent_goals)) == 1:
                # All 5 recent goals are identical → hard loop
                return EvaluationResult(
                    signal=EvalSignal.REPLAN,
                    reasoning=f"Agent has been stuck on the same goal for 5 steps: '{recent_goals[0][:50]}'",
                    new_plan="The current approach has completely stalled. Choose a fundamentally different strategy.",
                    failed_because=f"Repeated the same goal 5 times without progress: '{recent_goals[0][:60]}'",
                    alternatives=[
                        "Navigate to a different URL or section of the page",
                        "Use keyboard shortcuts or a different interaction method",
                        "Skip this sub-task and continue with the next part of the goal",
                    ],
                )

        # Check for circuit breaker signal in scratchpad
        circuit_breaker = any(
            "CIRCUIT BREAKER" in s or "LLM service appears to be down" in s
            for s in scratchpad[-3:]
        )
        if circuit_breaker:
            return EvaluationResult(
                signal=EvalSignal.ESCALATE,
                reasoning="LLM circuit breaker triggered — repeated API failures.",
                alert_message="LLM service is experiencing failures. Check API key and model availability.",
            )

        # Check if agent already called done() — if so, COMPLETE immediately
        if history_entries:
            last_entry = history_entries[-1]
            for result in last_entry.action_results:
                if result.action_name == "done" and result.success:
                    return EvaluationResult(
                        signal=EvalSignal.COMPLETE,
                        reasoning="Agent called done() — task completed.",
                        completion_summary=result.extracted_content or "Task completed.",
                    )

        # No clear heuristic decision → defer to LLM
        return None

    def _build_prompt(
        self,
        goal: str,
        stop_condition: StopCondition,
        stop_progress: str,
        stop_met: bool,
        recent_history: str,
        scratchpad: list[str],
        step_number: int,
        consecutive_failures: int,
        metrics: dict[str, float],
        elapsed_seconds: float,
    ) -> str:
        scratchpad_text = "\n".join(f"  - {s}" for s in scratchpad[-10:]) or "  (empty)"
        metrics_text = json.dumps(metrics, indent=2) if metrics else "  (none tracked)"

        return f"""You are an EvaluationAgent reviewing an autonomous agent's progress.

## Original Goal
{goal}

## Stop Condition
Type: {stop_condition.type}
Description: {stop_condition.description or 'N/A'}
Progress: {stop_progress}
Condition already met by metrics: {"YES" if stop_met else "NO"}

## Current State
Step number: {step_number}
Consecutive failures: {consecutive_failures}
Elapsed: {int(elapsed_seconds)}s

## Tracked Metrics
{metrics_text}

## Agent Scratchpad (key findings)
{scratchpad_text}

## Recent History (last 15 steps)
{recent_history}

## Your Job
Decide the appropriate signal:

- **CONTINUE**: The agent is making meaningful progress. Keep going.
- **COMPLETE**: The stop condition metric is met OR the goal is clearly achieved. Time to stop.
- **REPLAN**: The agent is stuck (same actions repeated, no progress for many steps, or approach is clearly wrong). Generate a better plan.
- **PAUSE**: The agent has hit a blocker that requires human input (e.g., CAPTCHA, missing credentials, ambiguous instructions).
- **ESCALATE**: Something is seriously wrong (crash loop, permissions error, unrecoverable state).

## Decision Rules
1. If stop_met == YES → strongly prefer COMPLETE unless the work is clearly unfinished
2. If consecutive_failures >= 5 → strongly prefer REPLAN
3. If same goal repeated in last 5 steps → prefer REPLAN
4. If agent is waiting for AI responses and making conversational progress → CONTINUE
5. If partial progress but stuck on one sub-task → REPLAN for that sub-task

Respond with ONLY valid JSON:
{{
  "signal": "continue" | "replan" | "complete" | "pause" | "escalate",
  "reasoning": "1-3 sentence explanation",
  "new_plan": "only if signal=replan: brief new direction in 1-2 sentences",
  "failed_because": "only if signal=replan: root cause in one sentence — WHY the current approach failed",
  "alternatives": ["only if signal=replan: most reliable alternative approach", "second option", "last resort fallback"],
  "completion_summary": "only if signal=complete: what was accomplished",
  "alert_message": "only if signal=pause or escalate: what the user needs to know"
}}"""

    async def _call_llm(self, prompt: str) -> EvaluationResult:
        """Call LLM with JSON mode fallback."""
        messages = [{"role": "user", "content": prompt}]

        async def _do(use_json_mode: bool) -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1024,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = await self.llm_client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("Empty LLM response")
            return content

        # Try with JSON mode first, fall back without
        try:
            raw = await _do(use_json_mode=True)
        except Exception:
            raw = await _do(use_json_mode=False)

        return self._parse(raw)

    def _parse(self, raw: str) -> EvaluationResult:
        """Parse LLM output into EvaluationResult."""
        text = raw.strip()

        # Strip markdown code fences
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        # Find outermost JSON object
        start = text.find("{")
        if start != -1:
            text = text[start:]
            end = text.rfind("}")
            if end != -1:
                text = text[: end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"EvaluationAgent JSON parse failed: {e} — defaulting to CONTINUE")
            return EvaluationResult(
                signal=EvalSignal.CONTINUE,
                reasoning="Could not parse evaluator response — continuing by default.",
            )

        # Normalise signal
        raw_signal = str(data.get("signal", "continue")).lower().strip()
        try:
            signal = EvalSignal(raw_signal)
        except ValueError:
            signal = EvalSignal.CONTINUE

        # Extract alternatives — ensure it's a list of strings
        raw_alts = data.get("alternatives")
        alternatives: list[str] | None = None
        if isinstance(raw_alts, list):
            alternatives = [str(a) for a in raw_alts if a]
        elif isinstance(raw_alts, str) and raw_alts.strip():
            alternatives = [raw_alts]

        return EvaluationResult(
            signal=signal,
            reasoning=str(data.get("reasoning", "")),
            new_plan=data.get("new_plan"),
            failed_because=data.get("failed_because"),
            alternatives=alternatives if alternatives else None,
            completion_summary=data.get("completion_summary"),
            alert_message=data.get("alert_message"),
        )

    def _summarize_history(self, entries: list[Any]) -> str:
        """Compact history summary for the evaluation prompt."""
        if not entries:
            return "  (no history yet)"
        lines = []
        for e in entries:
            ao = e.agent_output
            results_text = "; ".join(
                ("✅" if r.success else f"❌ {r.error[:60] if r.error else ''}").strip()
                for r in e.action_results
            )
            lines.append(
                f"  Step {e.step_number + 1}: {ao.next_goal[:80]}"
                f"\n    Actions: {results_text}"
            )
        return "\n".join(lines)
