"""
autobot/autonomy.py — Phased, indefinite autonomous runner.

Architecture (multi-agent loop):
  1. PLANNER: decompose_goal() → list of phases (run once at start)
  2. CONTEXT: summarize_page() → inject page summary into state before each loop
  3. EXECUTOR: decide_next_steps() → take concrete browser/desktop actions
  4. VERIFIER: verify_progress() → check if current phase / overall goal is done

The loop runs until:
  - Verifier says goal_done=True, OR
  - User calls cancel(), OR
  - max_hours is exceeded (optional safety cap, default 8h)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .engine import AutomationEngine, ExecutionResult, TaskStep
from .llm_brain import LLMBrain


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class AutonomousConfig:
    # Safety
    max_steps_per_loop: int = 5
    max_hours: float = 8.0           # hard-stop after this many hours (0 = no limit)

    # Diagnostics
    diagnostics_command: str = ""
    target_url: str = ""

    # Permissions
    allow_desktop_actions: bool = False
    allow_sensitive_adapter_actions: bool = False

    # Legacy compat (ignored but kept so callers don't break)
    max_loops: int = 0


# ── Runner ────────────────────────────────────────────────────────────────────

class AutonomousRunner:
    def __init__(self, engine: AutomationEngine, logger=None) -> None:
        self.engine = engine
        self.logger = logger or (lambda _msg: None)
        self.brain = LLMBrain(logger=self.logger)
        self._cancel_requested = False

        # Public status (readable from outside the thread)
        self.current_phase: str = ""
        self.current_phase_index: int = 0
        self.phase_plan: list[str] = []
        self.loop_count: int = 0
        self.is_running: bool = False

    def cancel(self) -> None:
        self._cancel_requested = True
        self.engine.cancel()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, goal: str, config: AutonomousConfig, page_context: dict[str, str] | None = None) -> ExecutionResult:
        """
        Run the autonomous agent until the goal is achieved or cancelled.

        page_context: optional {"url": ..., "title": ..., "text": ...}
                      from the browser extension; injected as initial state context.
        """
        self._cancel_requested = False
        self.is_running = True
        self.loop_count = 0
        self.engine.state["autonomy_goal"] = goal
        self.engine.state["autonomy_loops"] = 0

        deadline = time.time() + (config.max_hours * 3600) if config.max_hours > 0 else float("inf")

        # ── Global Mission Progress ──────────────────────────────────────────
        mission_progress = self.engine.knowledge.get("mission_progress") or "Mission just started."
        self.engine.state["mission_progress"] = mission_progress
        self.logger(f"[MISSION] Current progress: {mission_progress}")

        try:
            result = self._run_inner(goal, config, page_context or {}, deadline)
            
            # Update mission progress upon completion
            if result.success:
                summary = self.brain.summarize_page(
                    url="N/A",
                    title="Work Summary",
                    raw_text=f"Goal: {goal}\nLoops: {self.loop_count}\nFinal State: {list(self.engine.state.keys())}"
                )
                new_progress = f"Previously: {mission_progress}\nLast Session: {summary}"
                self.engine.knowledge.set("mission_progress", new_progress)
                self.logger("[MISSION] Progress updated in Knowledge Base.")
            
            return result
        finally:
            self.is_running = False

    def _run_inner(
        self,
        goal: str,
        config: AutonomousConfig,
        page_context: dict[str, str],
        deadline: float,
    ) -> ExecutionResult:
        allowed = _allowed_actions(config.allow_desktop_actions)

        # ── Phase 1: Planner — decompose goal ─────────────────────────────────
        context_summary = ""
        if page_context.get("url") or page_context.get("text"):
            context_summary = self.brain.summarize_page(
                url=page_context.get("url", ""),
                title=page_context.get("title", ""),
                raw_text=page_context.get("text", ""),
            )
            self.engine.state["page_text_summary"] = context_summary
            self.engine.state["current_url"] = page_context.get("url", "")
            self.logger(f"[CONTEXT] Page summary: {context_summary[:200]}")

        self.logger("[PLANNER] Decomposing goal into phases…")
        phases_plan = self.brain.decompose_goal(goal=goal, context=context_summary)
        self.phase_plan = phases_plan.phases
        self.engine.state["phase_plan"] = self.phase_plan
        self.logger(f"[PLANNER] {len(self.phase_plan)} phases: {self.phase_plan}")
        self.logger(f"[PLANNER] Reasoning: {phases_plan.reasoning}")

        # ── Phase 2: Execute each phase until goal complete ────────────────────
        for phase_idx, phase in enumerate(self.phase_plan):
            if self._cancel_requested:
                break
            if time.time() > deadline:
                self.logger("[TIMEOUT] Max hours reached. Stopping.")
                break

            self.current_phase_index = phase_idx
            self.current_phase = phase
            self.engine.state["current_phase"] = phase
            self.engine.state["current_phase_index"] = phase_idx
            self.logger(f"\n{'='*60}")
            self.logger(f"[PHASE {phase_idx + 1}/{len(self.phase_plan)}] {phase}")
            self.logger(f"{'='*60}")

            phase_done = self._execute_phase(
                goal=goal,
                phase=phase,
                phase_idx=phase_idx,
                config=config,
                allowed=allowed,
                deadline=deadline,
            )

            if self._cancel_requested:
                break

            if phase_done:
                self.logger(f"[PHASE {phase_idx + 1}] ✓ Complete")
            else:
                self.logger(f"[PHASE {phase_idx + 1}] Phase ended (cancelled or timeout)")

            # After the last phase, check overall goal
            if phase_idx == len(self.phase_plan) - 1:
                self.logger("[VERIFIER] Checking overall goal completion…")
                verdict = self.brain.verify_progress(goal=goal, current_phase=phase, state=self.engine.state)
                self.logger(f"[VERIFIER] goal_done={verdict.goal_done} | {verdict.feedback}")
                if verdict.goal_done:
                    self.logger("🎉 Goal achieved!")
                    self.engine.close()
                    return ExecutionResult(
                        success=True,
                        completed_steps=self.loop_count,
                        total_steps=self.loop_count,
                        state=dict(self.engine.state),
                    )

        self.engine.close()
        return ExecutionResult(
            success=self._cancel_requested is False,
            completed_steps=self.loop_count,
            total_steps=self.loop_count,
            state=dict(self.engine.state),
        )

    # ── Phase executor loop ───────────────────────────────────────────────────

    def _execute_phase(
        self,
        goal: str,
        phase: str,
        phase_idx: int,
        config: AutonomousConfig,
        allowed: list[str],
        deadline: float,
    ) -> bool:
        """Run executor + verifier loops until the phase is complete."""
        max_phase_loops = 20  # safety: don't spin forever on one phase
        phase_loop = 0

        while phase_loop < max_phase_loops:
            if self._cancel_requested or time.time() > deadline:
                return False

            self.loop_count += 1
            phase_loop += 1
            self.engine.state["autonomy_loops"] = self.loop_count
            self.logger(f"\n[LOOP {self.loop_count}] Phase {phase_idx + 1}: {phase}")

            # Diagnostics step at start of each loop
            diag_result = self._run_diagnostics(config)
            if not diag_result.success:
                self.logger("[DIAG] Diagnostics failed — continuing anyway")

            # Refresh page context if we have a browser active
            self._refresh_page_context()

            # Executor: decide what to do
            phase_goal = f"GOAL: {goal}\nCURRENT PHASE: {phase}"
            decision = self.brain.decide_next_steps(
                goal=phase_goal,
                state=self.engine.state,
                allowed_actions=allowed,
                max_steps=config.max_steps_per_loop,
            )
            decision.steps = _sanitize_decision_steps(decision.steps, config.allow_sensitive_adapter_actions)
            self.logger(f"[EXECUTOR] {decision.reason}")
            self.logger(f"[EXECUTOR] {len(decision.steps)} steps planned | done_flag={decision.done}")

            if decision.done:
                # Executor says done — verify
                verdict = self.brain.verify_progress(goal=goal, current_phase=phase, state=self.engine.state)
                self.logger(f"[VERIFIER] phase_complete={verdict.phase_complete} goal_done={verdict.goal_done} | {verdict.feedback}")
                if verdict.goal_done or verdict.phase_complete:
                    return True
                # Executor thought it was done but verifier disagrees — continue
                self.logger("[VERIFIER] Continuing — goal not yet verified complete")

            if not decision.steps:
                self.logger("[EXECUTOR] No steps returned — waiting and retrying")
                time.sleep(5)
                continue

            # Run steps
            exec_result = self.engine.run_steps(
                steps=decision.steps,
                plan_name=f"phase_{phase_idx + 1}_loop_{phase_loop}",
                plan_description=f"Phase {phase_idx + 1}: {phase}",
                close_on_finish=False,
            )

            if not exec_result.success:
                self.logger(f"[EXECUTOR] Step failed. Asking brain for recovery…")
                fallback = self.brain.decide_next_steps(
                    goal=f"{phase_goal}\n[RECOVERY] Previous action failed. last_error is in state. Suggest fallback (wait longer, different selector, or request_human_help for CAPTCHA).",
                    state=self.engine.state,
                    allowed_actions=allowed,
                    max_steps=3,
                )
                fallback_steps = _sanitize_decision_steps(fallback.steps, config.allow_sensitive_adapter_actions)
                if fallback_steps:
                    for s in fallback_steps:
                        s.continue_on_error = True
                    self.engine.run_steps(
                        steps=fallback_steps,
                        plan_name=f"recovery_{phase_idx + 1}_loop_{phase_loop}",
                        plan_description="Recovery after step failure",
                        close_on_finish=False,
                    )

            # After executing, check phase progress
            verdict = self.brain.verify_progress(goal=goal, current_phase=phase, state=self.engine.state)
            self.logger(f"[VERIFIER] phase_complete={verdict.phase_complete} | {verdict.feedback}")
            if verdict.goal_done:
                return True
            if verdict.phase_complete:
                return True

        self.logger(f"[PHASE] Max phase loops ({max_phase_loops}) reached — moving on")
        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_diagnostics(self, config: AutonomousConfig) -> ExecutionResult:
        steps: list[TaskStep] = []
        if config.target_url:
            steps.append(TaskStep(action="open_url", args={"url": config.target_url}, description="Open target URL", retries=1, continue_on_error=True))
            steps.append(TaskStep(action="wait", args={"seconds": 2.0}, description="Wait for page load"))
            steps.append(TaskStep(action="browser_read_console_errors", description="Capture console errors", save_as="console_errors", continue_on_error=True))
        if config.diagnostics_command:
            steps.append(TaskStep(
                action="run_command",
                args={"command": config.diagnostics_command, "timeout_seconds": 180},
                description=f"Run diagnostics: {config.diagnostics_command}",
                save_as="last_test_output",
                continue_on_error=True,
            ))
        if not steps:
            return ExecutionResult(success=True, completed_steps=0, total_steps=0, state=self.engine.state)
        return self.engine.run_steps(steps=steps, plan_name="diagnostics", close_on_finish=False)

    def _refresh_page_context(self) -> None:
        """Try to read the current page URL and text content; update state."""
        try:
            browser = self.engine.browser
            if hasattr(browser, "_page") and browser._page is not None and not browser._page.is_closed():
                url = browser._page.url
                title = browser._page.title()
                self.engine.state["current_url"] = url
                # Quick text grab (limited to avoid huge state)
                try:
                    raw_text = browser._page.inner_text("body")[:8000]
                    summary = self.brain.summarize_page(url=url, title=title, raw_text=raw_text)
                    self.engine.state["page_text_summary"] = summary
                    self.logger(f"[CONTEXT] {url[:80]} — {summary[:120]}")
                except Exception:
                    pass
        except Exception:
            pass

    def get_status(self) -> dict[str, Any]:
        return {
            "is_running": self.is_running,
            "loop_count": self.loop_count,
            "current_phase_index": self.current_phase_index,
            "current_phase": self.current_phase,
            "phase_plan": self.phase_plan,
            "cancel_requested": self._cancel_requested,
        }


# ── Utility functions ─────────────────────────────────────────────────────────

def _allowed_actions(allow_desktop_actions: bool) -> list[str]:
    core = [
        "log", "wait",
        "adapter_list_actions", "adapter_call", "adapter_set_policy",
        "adapter_prepare_sensitive", "adapter_confirm_sensitive", "adapter_get_telemetry",
        "open_url", "search_google",
        "browser_fill", "browser_click", "browser_press", "browser_scroll",
        "browser_snapshot", "browser_get_status", "browser_get_url", "browser_set_mode",
        "browser_read_text", "browser_read_console_errors",
        "request_human_help", "request_human_input",
        "run_command", "open_vscode", "open_app", "open_path",
        "clipboard_get", "clipboard_set",
        "knowledge_get", "knowledge_set", "knowledge_search",
    ]
    if allow_desktop_actions:
        core.extend(["desktop_type", "desktop_hotkey", "desktop_move", "desktop_click", "desktop_switch_window"])
    return core


def _sanitize_decision_steps(steps: list[TaskStep], allow_sensitive: bool) -> list[TaskStep]:
    sanitized: list[TaskStep] = []
    for step in steps:
        if step.action == "adapter_call":
            args = dict(step.args)
            if not allow_sensitive:
                args["confirmed"] = False
            step = TaskStep(
                action=step.action,
                args=args,
                save_as=step.save_as,
                description=step.description,
                condition=step.condition,
                retries=step.retries,
                retry_delay_seconds=step.retry_delay_seconds,
                continue_on_error=step.continue_on_error,
            )
        sanitized.append(step)
    return sanitized
