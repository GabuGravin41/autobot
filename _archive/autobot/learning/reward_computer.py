"""
Reward Computer — Converts step outcomes into scalar reward signals.

Reward design principles (from RLHF / robotics literature):
  - Shape rewards to guide BEHAVIOUR, not just final outcomes.
  - Negative rewards for wasted effort (loops, drift, errors) discourage bad habits.
  - Bonus for efficiency — reaching the goal in fewer steps is better.
  - Small progress rewards keep the agent moving forward.
  - Large terminal rewards (success / failure) anchor the value function.

Reward table
============
Outcome                              Reward
──────────────────────────────────────────
Action succeeded                     +0.5
Action succeeded and URL changed     +0.8  (navigation progress)
Action succeeded and goal words hit  +1.0  (semantic progress)
Task completed successfully          +5.0  (terminal)
Task failed / timed out             -2.0  (terminal)
Action failed (retryable)           -0.3
Action failed (permanent)           -0.8
Same goal repeated (loop)           -0.4
Coordinate drift detected           -0.5  (clicking same spot)
LLM circuit breaker triggered       -1.0
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardContext:
    """All the information needed to compute the reward for one step."""
    # Action outcome
    action_name: str
    action_params: dict
    success: bool
    error: str | None

    # State transitions
    url_before: str
    url_after: str
    goal: str
    current_goal: str           # agent's micro-goal for this step

    # Loop / drift indicators
    consecutive_failures: int
    same_goal_count: int        # how many steps had the same micro-goal
    coordinate_drift: bool
    llm_circuit_breaker: bool

    # Terminal conditions
    task_done: bool
    task_success: bool | None   # None = not terminal


class RewardComputer:
    """
    Stateless reward calculator.

    Call compute(ctx) → float after each step to get the reward signal.
    """

    # ── Reward constants ──────────────────────────────────────────────────────

    # Terminal
    TASK_SUCCESS      = +5.0
    TASK_FAILURE      = -2.0

    # Per-step action outcomes
    ACTION_SUCCESS    = +0.5
    NAV_PROGRESS      = +0.8    # success + URL changed (navigated somewhere new)
    SEMANTIC_HIT      = +1.0    # success + goal keywords in new URL/content
    ACTION_RETRY      = -0.3    # failed but retryable
    ACTION_PERMANENT  = -0.8    # permanent failure (permission denied, 404)

    # Behaviour shaping
    GOAL_LOOP         = -0.4    # same micro-goal repeated
    COORD_DRIFT       = -0.5    # coordinate drift
    CIRCUIT_BREAKER   = -1.0    # LLM circuit breaker triggered

    # Efficiency bonus
    EFFICIENCY_BONUS  = +0.2    # first-try success with no prior failures

    def compute(self, ctx: RewardContext) -> float:
        """
        Compute the reward for a single step.

        Applies rewards additively — multiple bonuses/penalties can stack.
        Clamps final reward to [-3.0, +6.0] to prevent extreme values.
        """
        reward = 0.0

        # ── Terminal conditions (dominate everything else) ────────────────────
        if ctx.task_done:
            if ctx.task_success:
                return self.TASK_SUCCESS
            else:
                return self.TASK_FAILURE

        # ── LLM circuit breaker ───────────────────────────────────────────────
        if ctx.llm_circuit_breaker:
            reward += self.CIRCUIT_BREAKER

        # ── Action outcome ────────────────────────────────────────────────────
        if ctx.success:
            url_changed = ctx.url_before != ctx.url_after and ctx.url_after
            if url_changed and _semantic_match(ctx.goal, ctx.url_after):
                reward += self.SEMANTIC_HIT
            elif url_changed:
                reward += self.NAV_PROGRESS
            else:
                reward += self.ACTION_SUCCESS

            # Efficiency bonus: succeeded with no prior consecutive failures
            if ctx.consecutive_failures == 0:
                reward += self.EFFICIENCY_BONUS
        else:
            error_type = _classify_error_severity(ctx.error)
            if error_type == "permanent":
                reward += self.ACTION_PERMANENT
            elif error_type == "transient":
                reward += self.ACTION_RETRY * 0.5   # softer penalty — not agent's fault
            else:
                reward += self.ACTION_RETRY

        # ── Behaviour penalties ───────────────────────────────────────────────
        if ctx.same_goal_count > 2:
            # Escalating penalty the longer a loop persists
            loop_penalty = self.GOAL_LOOP * min(ctx.same_goal_count - 1, 4)
            reward += loop_penalty

        if ctx.coordinate_drift:
            reward += self.COORD_DRIFT

        # ── Clamp and return ─────────────────────────────────────────────────
        return max(-3.0, min(6.0, reward))

    def compute_run_final(
        self,
        steps_taken: int,
        max_steps: int,
        success: bool,
        failure_rate: float,
    ) -> float:
        """
        End-of-run final reward.

        Separate from per-step rewards — used to back-propagate a terminal
        signal into the policy memory.

        Args:
            steps_taken:  how many steps the agent used
            max_steps:    the budget (used to compute efficiency)
            success:      whether the task was judged successful
            failure_rate: fraction of steps that failed (0.0 = perfect run)
        """
        if success:
            # Scale by efficiency: used 20% of budget = full bonus, 100% = half
            efficiency = 1.0 - (steps_taken / max(max_steps, 1)) * 0.5
            return round(self.TASK_SUCCESS * efficiency, 2)
        else:
            # Heavier penalty for high failure rate
            return round(self.TASK_FAILURE * (1.0 + failure_rate), 2)

    def label(self, reward: float) -> str:
        """Human-readable label for a reward value."""
        if reward >= 4.0:
            return "🏆 task success"
        if reward >= 1.0:
            return "✅ good progress"
        if reward >= 0.3:
            return "→ minor progress"
        if reward >= -0.2:
            return "— neutral"
        if reward >= -1.0:
            return "⚠️ bad step"
        return "❌ severe penalty"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _semantic_match(goal: str, url: str) -> bool:
    """Check if meaningful goal keywords appear in the new URL."""
    goal_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", goal.lower()))
    stop = {"search", "find", "open", "navigate", "click", "the", "and", "for"}
    goal_words -= stop
    url_lower = url.lower()
    return any(w in url_lower for w in goal_words)


def _classify_error_severity(error: str | None) -> str:
    """
    Return 'permanent', 'retryable', or 'transient' for an error string.

    permanent  → high penalty, agent should not retry the same action
    retryable  → moderate penalty, agent may try slight variation
    transient  → low penalty, likely network/timing issue — retry is fine
    """
    if not error:
        return "retryable"
    e = error.lower()

    # Permanent failures — retrying will definitely not help
    _PERMANENT = (
        "permission denied", "forbidden", "403", "404", "not found",
        "access denied", "unauthorized", "authentication failed",
        "invalid url", "no such file", "element not found",
        "selector not found", "does not exist",
    )
    if any(k in e for k in _PERMANENT):
        return "permanent"

    # Transient failures — almost certainly recoverable with a brief wait
    _TRANSIENT = (
        "timeout", "timed out", "connection refused", "network", "429",
        "rate limit", "resource exhausted", "temporarily unavailable",
        "service unavailable", "503",
    )
    if any(k in e for k in _TRANSIENT):
        return "transient"

    return "retryable"


# Module-level singleton
reward_computer = RewardComputer()
