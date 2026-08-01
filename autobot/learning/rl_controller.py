"""
RL Controller — Integrates the RL pipeline with the AgentLoop.

Called by AgentLoop at two points in each step:
  1. BEFORE the step: get_affordances_hint() → inject learned preferences into prompt
  2. AFTER the step:  record_step() → store experience, update policy

Also called at run end: record_run_end() → terminal reward signal.

The controller is a thin coordinator — it doesn't contain learning logic
itself. It delegates to:
  - ExperienceStore: raw (state, action, outcome) storage
  - RewardComputer: reward signal calculation
  - PolicyMemory: learned preferences per context

Design principle: every call must be NON-BLOCKING and fault-tolerant.
If the RL pipeline fails for any reason, the agent loop must not crash.
All exceptions are caught, logged, and silently ignored.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class RLController:
    """
    Thin coordinator between AgentLoop and the RL pipeline.

    One shared singleton for the process — all runs share the same
    ExperienceStore and PolicyMemory (so knowledge accumulates across runs).

    Each run gets a unique _run_id assigned by new_run(). Multiple concurrent
    runs are safe because the underlying SQLite store is thread-safe.

    Usage in AgentLoop:
        # At start of run:
        self._rl.new_run()

        # At start of each step (before LLM call):
        hint = self._rl.get_affordances_hint(url, goal)

        # After step execution:
        self._rl.record_step(...)

        # At end of run:
        self._rl.record_run_end(...)
    """

    def __init__(self) -> None:
        # Lazy-import to avoid circular imports and startup overhead
        self._store = None
        self._reward = None
        self._policy = None
        self._run_id = str(uuid.uuid4())[:8]
        self._save_interval = 10     # save policy every N steps
        self._step_counter = 0
        self._enabled = True

        # Track steps per (url, goal) for same-goal detection
        self._goal_counts: dict[str, int] = {}

    def _init(self) -> bool:
        """Lazy-initialize the RL pipeline. Returns False if init fails."""
        if self._store is not None:
            return self._enabled
        try:
            from autobot.learning.experience_store import ExperienceStore, _url_pattern, _task_keywords, _action_tool, StepState
            from autobot.learning.reward_computer import RewardComputer, RewardContext
            from autobot.learning.policy_memory import PolicyMemory
            self._store = ExperienceStore()
            self._reward = RewardComputer()
            self._policy = PolicyMemory()
            self._url_pattern = _url_pattern
            self._task_keywords = _task_keywords
            self._action_tool = _action_tool
            self._StepState = StepState
            self._RewardContext = RewardContext
            logger.info(
                f"🧠 RL pipeline initialized | "
                f"{self._store.total_experiences()} experiences loaded | "
                f"{self._policy.summary()['contexts']} learned contexts"
            )
            return True
        except Exception as e:
            logger.warning(f"RL pipeline init failed (running without RL): {e}")
            self._enabled = False
            return False

    # ── Pre-step: prompt enrichment ───────────────────────────────────────────

    def get_affordances_hint(self, url: str, goal: str) -> str | None:
        """
        Return a learned affordances hint to inject into the agent's prompt.

        This tells the agent which tools historically work best on this page.
        Returns None if no learned data exists yet (first few runs).

        Called BEFORE each LLM call — must be fast (in-memory lookup).
        """
        if not self._init():
            return None
        try:
            url_pattern = self._url_pattern(url)
            page_type = _infer_page_type_from_url(url)
            task_kw = self._task_keywords(goal)
            return self._policy.build_affordances_hint(url_pattern, page_type, task_kw)
        except Exception as e:
            logger.debug(f"get_affordances_hint failed (non-fatal): {e}")
            return None

    # ── Post-step: experience recording ───────────────────────────────────────

    def record_step(
        self,
        *,
        url: str,
        goal: str,
        action_name: str,
        action_params: dict,
        success: bool,
        error: str | None,
        step_number: int,
        url_before: str,
        url_after: str,
        current_goal: str = "",
        consecutive_failures: int = 0,
        same_goal_count: int = 0,
        coordinate_drift: bool = False,
        llm_circuit_breaker: bool = False,
        task_done: bool = False,
        task_success: bool | None = None,
    ) -> float | None:
        """
        Record a completed step and update the policy.

        Returns the computed reward (for dashboard display) or None on error.
        Always non-blocking and fault-tolerant.
        """
        if not self._init():
            return None
        try:
            self._step_counter += 1

            # ── Compute reward ────────────────────────────────────────────────
            ctx = self._RewardContext(
                action_name=action_name,
                action_params=action_params,
                success=success,
                error=error,
                url_before=url_before,
                url_after=url_after,
                goal=goal,
                current_goal=current_goal,
                consecutive_failures=consecutive_failures,
                same_goal_count=same_goal_count,
                coordinate_drift=coordinate_drift,
                llm_circuit_breaker=llm_circuit_breaker,
                task_done=task_done,
                task_success=task_success,
            )
            reward = self._reward.compute(ctx)

            # ── Record in experience store ────────────────────────────────────
            state = self._StepState(
                url=url,
                goal=goal,
                action_name=action_name,
                action_params=action_params,
                success=success,
                error=error,
                step_number=step_number,
                run_id=self._run_id,
            )
            self._store.record(state, reward)

            # ── Update policy memory ──────────────────────────────────────────
            url_pattern = self._url_pattern(url)
            page_type = _infer_page_type_from_url(url)
            task_kw = self._task_keywords(goal)
            action_tool = self._action_tool(action_name, action_params)

            self._policy.update(
                url_pattern=url_pattern,
                page_type=page_type,
                task_kw=task_kw,
                action_tool=action_tool,
                success=success,
                reward=reward,
            )

            # ── Periodic policy save ──────────────────────────────────────────
            if self._step_counter % self._save_interval == 0:
                self._policy.save()

            label = self._reward.label(reward)
            logger.debug(
                f"RL step {step_number}: {action_name} → {label} "
                f"(reward={reward:+.2f}, success={success})"
            )
            return reward

        except Exception as e:
            logger.debug(f"record_step failed (non-fatal): {e}")
            return None

    def record_run_end(
        self,
        *,
        goal: str,
        steps: int,
        max_steps: int,
        success: bool,
        failure_rate: float = 0.0,
    ) -> None:
        """
        Record the terminal reward signal at end of run and save the policy.

        Called by runner.py after the agent loop finishes.
        """
        if not self._init():
            return
        try:
            terminal_reward = self._reward.compute_run_final(
                steps_taken=steps,
                max_steps=max(max_steps, 1),
                success=success,
                failure_rate=failure_rate,
            )
            label = self._reward.label(terminal_reward)
            logger.info(
                f"🎯 Run {self._run_id} complete: {label} "
                f"(terminal_reward={terminal_reward:+.2f}, success={success}, "
                f"steps={steps}/{max_steps})"
            )

            # Save policy with terminal signal context
            self._policy.save()

            # Log run summary
            summary = self._store.recent_run_summary(self._run_id)
            if summary:
                logger.info(
                    f"📊 Run summary: {summary.get('total', 0)} steps, "
                    f"{summary.get('successes', 0)} successes, "
                    f"avg_reward={summary.get('avg_reward', 0):.2f}"
                )
        except Exception as e:
            logger.debug(f"record_run_end failed (non-fatal): {e}")

    def get_stats(self) -> dict:
        """Return RL pipeline stats for the dashboard."""
        if not self._init():
            return {"enabled": False}
        try:
            return {
                "enabled": True,
                "run_id": self._run_id,
                "total_experiences": self._store.total_experiences(),
                "policy_summary": self._policy.summary(),
                "step_counter": self._step_counter,
            }
        except Exception:
            return {"enabled": True, "error": "stats unavailable"}

    def new_run(self) -> None:
        """Start a fresh run ID. Call at the beginning of each agent run."""
        self._run_id = str(uuid.uuid4())[:8]
        self._goal_counts.clear()
        logger.debug(f"RL Controller: new run {self._run_id}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_page_type_from_url(url: str) -> str:
    """Fast URL-based page type inference (no DOM needed)."""
    url_lower = url.lower()
    if any(k in url_lower for k in ["login", "signin", "auth"]):
        return "auth"
    if any(k in url_lower for k in ["github.com", "gitlab.com"]):
        return "code_repo"
    if any(k in url_lower for k in ["leetcode", "codeforces", "hackerrank"]):
        return "coding_challenge"
    if any(k in url_lower for k in ["kaggle.com"]):
        return "data_platform"
    if any(k in url_lower for k in ["google.com/search", "bing.com/search"]):
        return "search"
    return "general"


# Module-level singleton — shared across the process
rl_controller = RLController()
