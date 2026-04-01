"""
Agent Loop — The core observe → think → act → verify cycle.

This is the new agent loop that wires together:
- DOM extraction (from Browser Use patterns)
- Prompt building (system prompt + step prompt with browser state)
- LLM calls for structured output (thinking/eval/memory/goal/actions)
- Action execution (click by index, input by index, navigate)
- Page change detection (skip remaining actions on navigation)
- Step history tracking

This replaces the original autonomy.py's _execute_phase() with a
loop that follows Browser Use's proven architecture.

Usage:
    agent = AgentLoop(page=page, llm_client=client, goal="search for AI papers")
    result = await agent.run()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from autobot.agent.models import (
    ActionModel,
    ActionResult,
    AgentOutput,
    AgentStepInfo,
    ClickAction,
    ComputerCallAction,
    DoneAction,
    InputTextAction,
    NavigateAction,
    PressKeyAction,
    ScrollAction,
    StepHistoryEntry,
)
from autobot.agent.approval import ApprovalGuard, RiskTier
from autobot.agent.evaluator import EvalSignal, EvaluationAgent
from autobot.agent.resource_manager import screen_lock
from autobot.agent.stop_condition import StopCondition, after_steps
from autobot.computer.computer import Computer
from autobot.dom.models import (
    BrowserState,
    DOMSerializedState,
    SelectorMap,
    TabInfo,
)
from autobot.prompts.builder import StepPromptBuilder, SystemPromptBuilder

logger = logging.getLogger(__name__)


def _strip_images_from_messages(messages: list[dict]) -> list[dict]:
    """Remove base64 image parts from messages for a text-only fallback retry."""
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [p for p in content if p.get("type") != "image_url"]
            if not text_parts:
                text_parts = [{"type": "text", "text": "[Screenshot unavailable — act based on goal and history]"}]
            result.append({**msg, "content": text_parts})
        else:
            result.append(msg)
    return result


class AgentLoop:
    """
    The core agent loop: observe → think → act → verify.

    Adapted from Browser Use's Agent class (agent/service.py).
    Each iteration:
        1. OBSERVE: Extract DOM tree + take screenshot
        2. THINK: Send state to LLM, get structured output
        3. ACT: Execute actions (stop on page change)
        4. RECORD: Save step to history for next iteration
    """

    def __init__(
        self,
        page: Any,  # Playwright Page
        llm_client: Any,  # OpenAI-compatible client
        goal: str,
        model: str = "gpt-4o",
        fast_model: str | None = None,  # Cheaper model for routine steps (multi-LLM routing)
        max_steps: int | None = 100,   # None = perpetual
        max_actions_per_step: int = 5,
        use_vision: bool = True,
        custom_instructions: str | None = None,
        stop_condition: StopCondition | None = None,
        task_id: str | None = None,    # For ScreenLock identification
    ):
        self.page = page
        self.llm_client = llm_client
        self.goal = goal
        self.model = model
        # fast_model: used for routine steps to reduce cost/latency.
        # Step 0 (planning) and REPLAN steps always use the primary model.
        self.fast_model: str | None = fast_model or os.getenv("AUTOBOT_FAST_MODEL")
        self.max_actions_per_step = max_actions_per_step
        self.use_vision = use_vision
        self.custom_instructions = custom_instructions

        # Stop condition — governs when the run should halt.
        # Env override: AUTOBOT_MAX_STEPS=0 means perpetual.
        env_steps = os.getenv("AUTOBOT_MAX_STEPS")
        if stop_condition is not None:
            self.stop_condition = stop_condition
            # Derive max_steps from stop_condition for backwards compat
            self.max_steps: int | None = stop_condition.max_steps if stop_condition.type == "steps" else None
        elif env_steps is not None:
            n = int(env_steps)
            self.max_steps = n if n > 0 else None
            self.stop_condition = after_steps(n) if n > 0 else StopCondition(type="none", description="Perpetual (env)")
        else:
            self.max_steps = max_steps  # may be None for perpetual
            self.stop_condition = after_steps(max_steps) if max_steps else StopCondition(type="none", description="Perpetual")

        # State
        self.step_number = 0
        self.history: list[StepHistoryEntry] = []
        self.previous_dom_state: DOMSerializedState | None = None
        self._run_start_time = time.time()

        # Scratchpad: persistent notes that accumulate across the entire run
        self.scratchpad: list[str] = []

        # Tracked metrics — EvaluationAgent checks these against stop_condition
        self.metrics: dict[str, float] = {}

        # Retry / failure tracking
        self._consecutive_failures = 0
        self._consecutive_step_errors = 0   # raw exception count
        self._consecutive_llm_failures = 0  # circuit breaker: LLM failures across steps
        self._last_goal = ""

        # URL loop detection: counts how many times each URL has been the active page
        self._url_visit_counts: dict[str, int] = {}
        self._url_loop_alerted: set[str] = set()  # URLs we've already warned about

        # Click coordinate tracking for drift detection
        self._recent_clicks: list[tuple[int, int]] = []  # last N click coordinates

        # Watchdog: track last time the screen changed (action had effect)
        self._last_progress_time = time.time()
        self._watchdog_seconds = int(os.getenv("AUTOBOT_WATCHDOG_SECONDS", "600"))  # 10 min

        # Authentication tracking
        self.pending_auth: dict | None = None

        # Evaluation: call EvaluationAgent every N steps
        self._eval_interval = int(os.getenv("AUTOBOT_EVAL_INTERVAL", "10"))
        self._last_eval_signal: str = "continue"
        self._evaluator: EvaluationAgent | None = (
            EvaluationAgent(llm_client=llm_client, model=model) if llm_client else None
        )

        # Checkpointing
        self._checkpoint_interval = int(os.getenv("AUTOBOT_CHECKPOINT_INTERVAL", "5"))
        self._run_dir: Path | None = None  # Set by runner if available

        # ScreenLock identity — used to identify this task in the resource manager
        self._task_id = task_id or f"loop-{id(self)}"

        # Window hint for smooth context switching: when this task re-acquires the
        # ScreenLock after another task used it, the hint tells us which window to
        # bring to front (e.g. "Google Chrome", "code", "Terminal").
        # Updated automatically on navigate actions and computer.display.focus() calls.
        self._window_hint: str = "Google Chrome"   # default: most tasks start in browser
        screen_lock.register_window(self._task_id, self._window_hint)

        # Approval guard — gates risky actions based on AUTOBOT_APPROVAL_MODE
        self._approval_guard = ApprovalGuard()

        # Cancellation: set to True to stop the loop ASAP (checked each iteration
        # and after each LLM call so blocking API calls also get interrupted).
        self._cancelled = False
        self._current_llm_task: asyncio.Task | None = None   # active LLM coroutine task

        # Pause: when True the loop sits idle between steps without consuming LLM calls.
        # resume() clears this flag and execution picks up at the next step boundary.
        self._paused = False

        # Smart vision skipping: track what the last actions were and whether screen changed.
        # Non-visual steps (typing, clipboard ops, terminal) skip sending the screenshot to the LLM.
        self._last_action_types: list[str] = []      # e.g. ["keyboard.type", "clipboard.set"]
        self._last_screenshot_hash: str | None = None  # MD5 of previous screenshot bytes

        # Click zoom: after each mouse click, store a 300×300 region crop for next step
        self._last_click_zoom_b64: str | None = None
        self._last_click_coords: tuple[int, int] | None = None

        # Persistent memory store — facts survive across runs
        from autobot.memory.store import memory_store
        self._memory_store = memory_store

        # RL pipeline — learn from step outcomes across runs
        try:
            from autobot.learning.rl_controller import rl_controller
            self._rl = rl_controller
            self._rl.new_run()
        except Exception as _rl_err:
            logger.debug(f"RL pipeline unavailable (non-fatal): {_rl_err}")
            self._rl = None

        # Prompt evolver — dynamic corrections based on trajectory
        try:
            from autobot.prompts.prompt_evolver import PromptEvolver
            self._evolver = PromptEvolver(goal=goal)
        except Exception as _pe_err:
            logger.debug(f"PromptEvolver unavailable (non-fatal): {_pe_err}")
            self._evolver = None

        # Computer API for OS-level tools
        self.computer = Computer()

        # Build system prompt
        tool_catalog = self.computer.get_tool_catalog()
        self.system_prompt_builder = SystemPromptBuilder(
            max_actions_per_step=max_actions_per_step,
            custom_instructions=custom_instructions,
            tool_catalog=tool_catalog,
        )
        self.system_prompt = self.system_prompt_builder.build()

        self.last_screenshot_path: str | None = None
        self._last_narrative: str = ""   # Latest plain-English "what I'm doing"

    def cancel(self) -> None:
        """Cancel the agent loop immediately.

        Sets the cancellation flag (checked at every loop iteration) AND cancels
        any in-flight LLM asyncio Task so the agent stops within milliseconds
        rather than waiting for a 60-90s API call to time out.
        """
        self._cancelled = True
        if self._current_llm_task and not self._current_llm_task.done():
            self._current_llm_task.cancel()
        logger.info("🛑 AgentLoop cancelled")

    def pause(self) -> None:
        """Pause execution after the current step completes.

        The loop stays alive but idles (checking every 0.3s) without making
        any LLM calls or taking actions. Call resume() to continue.
        """
        self._paused = True
        logger.info("⏸ AgentLoop paused")

    def resume(self) -> None:
        """Resume a paused agent loop."""
        self._paused = False
        logger.info("▶ AgentLoop resumed")

    async def _wait_if_paused(self) -> None:
        """Block until unpaused. Also respects cancellation."""
        while self._paused:
            if self._cancelled:
                return
            await asyncio.sleep(0.3)

    def get_status(self) -> dict[str, Any]:
        """Returns the current status metadata for the dashboard."""
        elapsed = time.time() - self._run_start_time
        return {
            "current_step": self.step_number,
            "max_steps": self.max_steps,
            "goal": self.goal,
            "last_screenshot_path": self.last_screenshot_path,
            "stop_condition": self.stop_condition.model_dump(),
            "stop_progress": self.stop_condition.progress_text({
                "step_number": self.step_number,
                "metrics": self.metrics,
                "elapsed_seconds": elapsed,
            }),
            "eval_signal": self._last_eval_signal,
            "metrics": self.metrics,
            "elapsed_seconds": int(elapsed),
            "narrative": self._last_narrative,
            "paused": self._paused,
        }

    def _save_checkpoint(self):
        """Save current progress to disk so the run survives crashes."""
        if not self._run_dir:
            return
        try:
            checkpoint = {
                "step_number": self.step_number,
                "goal": self.goal,
                "scratchpad": self.scratchpad,
                "metrics": self.metrics,
                "consecutive_failures": self._consecutive_failures,
                "last_goal": self._last_goal,
                "eval_signal": self._last_eval_signal,
                "elapsed_seconds": int(time.time() - self._run_start_time),
                "stop_condition": self.stop_condition.model_dump(),
                "max_steps": self.max_steps,
                "history_count": len(self.history),
                "history_summary": [
                    {
                        "step": e.step_number,
                        "goal": e.agent_output.next_goal,
                        "actions": [a.model_dump() for a in e.agent_output.action],
                        "results": [r.extracted_content or "" for r in e.action_results],
                    }
                    for e in self.history[-20:]
                ],
            }
            checkpoint_path = self._run_dir / "checkpoint.json"
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, default=str))
        except Exception as e:
            logger.debug(f"Checkpoint save failed (non-fatal): {e}")

    def _build_sc_context(self) -> dict[str, Any]:
        """Build the context dict used by StopCondition.is_met()."""
        return {
            "step_number": self.step_number,
            "metrics": self.metrics,
            "elapsed_seconds": time.time() - self._run_start_time,
        }

    async def _run_evaluation(self) -> EvalSignal:
        """
        Call EvaluationAgent to assess progress and get a signal.
        Returns CONTINUE on any failure.
        """
        if self._evaluator is None:
            return EvalSignal.CONTINUE
        try:
            result = await self._evaluator.evaluate(
                goal=self.goal,
                stop_condition=self.stop_condition,
                history_entries=self.history,
                scratchpad=self.scratchpad,
                step_number=self.step_number,
                consecutive_failures=self._consecutive_failures,
                metrics=self.metrics,
                elapsed_seconds=time.time() - self._run_start_time,
            )
            self._last_eval_signal = result.signal.value
            logger.info(f"📊 EvaluationAgent: {result.signal.value.upper()} — {result.reasoning}")

            if result.signal == EvalSignal.REPLAN and result.new_plan:
                # Inject new plan into scratchpad so agent sees it next step
                self.scratchpad.append(f"[REPLAN] {result.new_plan}")
                logger.info(f"🔄 New plan injected: {result.new_plan}")

            if result.signal in (EvalSignal.PAUSE, EvalSignal.ESCALATE):
                # Surface to frontend
                self.pending_auth = {
                    "url": getattr(self.page, "url", ""),
                    "type": result.signal.value,
                    "message": result.alert_message or result.reasoning,
                }

            return result.signal
        except Exception as e:
            logger.warning(f"EvaluationAgent call failed: {e}")
            return EvalSignal.CONTINUE

    async def run(self) -> str:
        """
        Run the agent loop.

        Supports three modes:
          - Bounded (max_steps set): runs up to max_steps, then evaluates
          - Perpetual (max_steps=None): runs until agent calls done() or metric met
          - EvaluationAgent guided: every _eval_interval steps, EvaluationAgent
            can signal COMPLETE, REPLAN, PAUSE, or ESCALATE

        Returns:
            The final result text from the done action, or a summary.
        """
        mode_str = f"max {self.max_steps} steps" if self.max_steps else "perpetual"
        logger.info(f"🤖 Agent starting: '{self.goal}' ({mode_str})")
        self._run_start_time = time.time()
        self.stop_condition.start_timer()
        # Register initial window hint so the ScreenLock can restore focus
        screen_lock.register_window(self._task_id, self._window_hint)

        while True:
            # ── Check cancellation ──────────────────────────────────────────
            if self._cancelled:
                logger.info("🛑 Agent loop stopped — task cancelled by user")
                self._save_checkpoint()
                return "Task cancelled."

            # ── Check pause ─────────────────────────────────────────────────
            await self._wait_if_paused()

            # ── Check step budget ───────────────────────────────────────────
            if self.max_steps is not None and self.step_number >= self.max_steps:
                await self._wait_if_paused()
                logger.info(f"📊 Step budget ({self.max_steps}) reached — consulting EvaluationAgent...")
                signal = await self._run_evaluation()
                if signal == EvalSignal.COMPLETE:
                    self._save_checkpoint()
                    return self._summarize_history() + f"\n\n[EvaluationAgent: COMPLETE] {self._last_eval_signal}"
                elif signal == EvalSignal.REPLAN:
                    # Grant extra steps for the new plan
                    extension = 50
                    self.max_steps += extension
                    logger.info(f"🔄 Step budget extended by {extension} for replanning (now {self.max_steps})")
                else:
                    # Any other signal at budget exhaustion = stop
                    logger.warning(f"⚠️ Step budget exhausted (signal={signal.value})")
                    self._save_checkpoint()
                    return self._summarize_history()

            # ── Check stop condition ────────────────────────────────────────
            sc_ctx = self._build_sc_context()
            if self.stop_condition.is_met(sc_ctx):
                await self._wait_if_paused()
                logger.info(f"🏁 Stop condition met: {self.stop_condition.progress_text(sc_ctx)}")
                signal = await self._run_evaluation()
                if signal == EvalSignal.CONTINUE:
                    # EvaluationAgent disagrees — keep going briefly
                    if self.max_steps is not None:
                        self.max_steps += 20
                    logger.info("EvaluationAgent says CONTINUE despite stop condition — extending run")
                else:
                    self._save_checkpoint()
                    summary = self._summarize_history()
                    return summary + f"\n\n[Stop condition met: {self.stop_condition.progress_text(sc_ctx)}]"

            # ── Watchdog ────────────────────────────────────────────────────
            idle_secs = time.time() - self._last_progress_time
            if idle_secs > self._watchdog_seconds:
                logger.warning(f"🐕 Watchdog: no progress in {int(idle_secs)}s")
                self.scratchpad.append(
                    f"[WATCHDOG] No screen change detected in {int(idle_secs / 60)} minutes. "
                    "Try a completely different approach."
                )
                self._last_progress_time = time.time()  # Reset to avoid spam

            # ── Execute step ────────────────────────────────────────────────
            await self._wait_if_paused()
            try:
                result = await self._execute_step()

                if result is not None:
                    logger.info(f"✅ Agent done at step {self.step_number + 1}: {result[:120]}")
                    self._save_checkpoint()
                    return result

                self.step_number += 1
                self._consecutive_step_errors = 0  # Reset on clean step

                # Periodic checkpoint
                if self.step_number % self._checkpoint_interval == 0:
                    self._save_checkpoint()

                # Periodic history compression (every 25 steps, when history > 15 entries)
                _compress_interval = int(os.getenv("AUTOBOT_COMPRESS_INTERVAL", "25"))
                if self.step_number > 0 and self.step_number % _compress_interval == 0 and len(self.history) > 15:
                    await self._compress_history()

                # Periodic EvaluationAgent check
                if self.step_number > 0 and self.step_number % self._eval_interval == 0:
                    await self._wait_if_paused()
                    signal = await self._run_evaluation()
                    if signal == EvalSignal.COMPLETE:
                        self._save_checkpoint()
                        return self._summarize_history() + "\n\n[EvaluationAgent: Goal achieved]"
                    elif signal == EvalSignal.PAUSE:
                        logger.info("⏸️ EvaluationAgent: PAUSE — waiting for user")
                        # Loop continues; frontend will surface the pending_auth alert

            except asyncio.CancelledError:
                # LLM task was cancelled via self.cancel() — exit cleanly
                logger.info("🛑 Step interrupted by cancellation")
                self._save_checkpoint()
                return "Task cancelled."

            except Exception as e:
                self._consecutive_step_errors += 1
                logger.error(f"❌ Step {self.step_number + 1} error ({self._consecutive_step_errors} consecutive): {e}")

                if self._consecutive_step_errors >= 5:
                    logger.error("🚨 5 consecutive step errors — emergency pause")
                    self.pending_auth = {
                        "url": getattr(self.page, "url", ""),
                        "type": "escalate",
                        "message": f"Agent hit 5 consecutive errors. Last: {e}. Human review needed.",
                    }
                    self._save_checkpoint()
                    await asyncio.sleep(5)  # Brief pause before continuing

                self.step_number += 1
                self._save_checkpoint()  # Save state after every error
                await asyncio.sleep(1)   # Small pause to avoid tight crash loops

    async def _execute_step(self) -> str | None:
        """
        Execute one step of the agent loop.

        Returns:
            Result text if the agent called "done", None otherwise.
        """
        step_start = time.time()

        # ─── 1. OBSERVE (Hybrid: DOM snapshot + Screenshot) ───
        logger.debug(f"Step {self.step_number + 1}: Capturing observation...")

        import base64
        from autobot.dom.page_snapshot import get_page_snapshot

        # Run screenshot + DOM snapshot in parallel to save time
        screenshot_bytes, page_snapshot = await asyncio.gather(
            self.page.screenshot(),
            get_page_snapshot(),
            return_exceptions=True,
        )

        # Handle any exceptions from parallel gather
        if isinstance(screenshot_bytes, Exception):
            logger.warning(f"Screenshot failed: {screenshot_bytes}")
            screenshot_bytes = b""
        if isinstance(page_snapshot, Exception):
            logger.debug(f"Page snapshot failed: {page_snapshot}")
            page_snapshot = None

        if page_snapshot:
            logger.debug(
                f"DOM snapshot: {page_snapshot.num_interactive} interactive elements, "
                f"{len(page_snapshot.text)} chars of text"
            )

        # Get screen resolution for coordinate guidance in prompt
        try:
            screen_w, screen_h = self.computer.display.size()
        except Exception:
            screen_w, screen_h = 1920, 1080

        # Save full-res screenshot for dashboard live-view
        try:
            from pathlib import Path
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            screenshot_path = screenshot_dir / "latest.png"
            screenshot_path.write_bytes(screenshot_bytes)
            self.last_screenshot_path = str(screenshot_path.absolute())
        except Exception as e:
            logger.warning(f"Failed to save agent screenshot: {e}")

        # Compress screenshot PNG → JPEG to save tokens
        llm_screenshot_b64 = self._compress_screenshot(screenshot_bytes)

        # Smart vision skipping: don't send the screenshot to the LLM if the last actions
        # were purely non-visual (typing, clipboard, terminal) AND the screen hasn't changed.
        # The screenshot is still saved to disk for the dashboard — only suppressed from LLM.
        import hashlib as _hashlib
        _curr_hash = _hashlib.md5(screenshot_bytes).hexdigest() if screenshot_bytes else None
        _skip_vision = False
        _NON_VISUAL_PREFIXES = (
            "keyboard.", "terminal.run", "terminal.start",
            "clipboard.set", "clipboard.copy", "wait",
        )
        if os.getenv("AUTOBOT_SMART_VISION", "1") == "1" and self._last_action_types:
            _last_nonvisual = all(
                any(a.startswith(pfx) for pfx in _NON_VISUAL_PREFIXES)
                for a in self._last_action_types
            )
            _screen_same = _curr_hash is not None and _curr_hash == self._last_screenshot_hash
            if _last_nonvisual and _screen_same:
                _skip_vision = True
                logger.debug(
                    f"Vision skip: last actions={self._last_action_types}, screen unchanged"
                )
        self._last_screenshot_hash = _curr_hash
        # For local/Ollama models on CPU, vision is extremely slow (no GPU).
        # If AUTOBOT_LOCAL_NO_VISION=1, disable vision entirely and rely on DOM snapshots.
        # DOM snapshots are actually MORE precise for clicking (exact coordinates vs guessing).
        _local_no_vision = os.getenv("AUTOBOT_LOCAL_NO_VISION", "0") == "1"
        if _local_no_vision:
            _skip_vision = True

        _use_vision_this_step = self.use_vision and not _skip_vision

        # Use real URL/title from DOM snapshot when available
        page_url = (page_snapshot.url if page_snapshot and page_snapshot.url else self.page.url)
        page_title = (
            page_snapshot.title
            if page_snapshot and page_snapshot.title
            else f"Human Mode | Screen: {screen_w}×{screen_h}"
        )

        # Build tab list from open HumanMode pages (tab_index = Chrome tab number)
        _open_tabs: list[TabInfo] = []
        try:
            if hasattr(self.page, 'context') and hasattr(self.page.context, 'pages'):
                for _p in self.page.context.pages:
                    _open_tabs.append(TabInfo(
                        tab_id=str(_p.tab_index),
                        url=_p.url or "about:blank",
                        title=f"Tab {_p.tab_index}",
                    ))
        except Exception:
            pass

        browser_state = BrowserState(
            url=page_url,
            title=f"{page_title} | Screen: {screen_w}×{screen_h}",
            tabs=_open_tabs,
            screenshot_b64=llm_screenshot_b64,
            element_tree=None,
            selector_map=SelectorMap(),
            num_links=page_snapshot.num_links if page_snapshot else 0,
            num_interactive=page_snapshot.num_interactive if page_snapshot else 0,
            total_elements=page_snapshot.num_interactive if page_snapshot else 0,
            page_info=None
        )

        url_before = browser_state.url

        # Extract native UI (Windows-only; silently skipped on Linux/Mac)
        native_ui = None
        if hasattr(self.computer, "window"):
            try:
                native_ui = self.computer.window.extract_ui()
            except Exception as e:
                logger.debug(f"Native UI extraction skipped: {e}")

        # Update previous state for new-element detection
        self.previous_dom_state = DOMSerializedState(
            element_tree=browser_state.element_tree,
            selector_map=browser_state.selector_map,
        )

        # Popup detection — inject scratchpad alert so agent handles it first
        if page_snapshot and page_snapshot.has_popup:
            popup_info = "; ".join(
                f'"{d.get("title","?")}\" [{"/".join(d.get("buttons",[]))}]'
                for d in page_snapshot.dialogs
            )
            alert = f"[POPUP] Dialog detected: {popup_info}. Handle this before continuing."
            if not any("[POPUP]" in s for s in self.scratchpad[-3:]):
                self.scratchpad.append(alert)
                logger.info(f"🔔 Popup detected: {popup_info}")

        # URL loop detection — if visiting the same URL too many times, push agent to try something different
        _url_key = page_url.split("?")[0].rstrip("/")  # normalise (strip query params & trailing slash)
        self._url_visit_counts[_url_key] = self._url_visit_counts.get(_url_key, 0) + 1
        _url_visits = self._url_visit_counts[_url_key]
        if _url_visits >= 4 and _url_key not in self._url_loop_alerted:
            self._url_loop_alerted.add(_url_key)
            self.scratchpad.append(
                f"[URL LOOP] You have been on '{_url_key}' {_url_visits} times without completing the goal. "
                "Try a completely different approach: a different URL, a different tool, or a different strategy. "
                "Do NOT keep repeating the same action."
            )
            logger.warning(f"🔁 URL loop detected: {_url_key} visited {_url_visits}x")

        # ─── 2. THINK ───
        logger.debug(f"Step {self.step_number + 1}: Thinking...")

        # Recall relevant memories for this goal
        recalled = self._memory_store.recall(self.goal, top_k=6)

        # Prompt evolution: get trajectory correction hint (if agent is drifting)
        _evolution_hint = None
        try:
            if self._evolver is not None:
                _recent_acts = [
                    {
                        "action_name": a.action_name,
                        "success": r.success,
                        "error": r.error or "",
                    }
                    for e in self.history[-5:]
                    for a, r in zip(e.agent_output.action, e.action_results)
                ]
                _evo = self._evolver.get_evolution_hint(
                    step_number=self.step_number,
                    url=url_before,
                    goal=self.goal,
                    recent_actions=_recent_acts,
                    consecutive_failures=self._consecutive_failures,
                    history=self.history[-5:],
                    max_steps=self.max_steps,
                )
                if _evo and _evo.type != "none":
                    _evolution_hint = _evo.text
                    logger.info(f"🧭 PromptEvolver [{_evo.type}] ({_evo.reason}): injecting correction")
        except Exception as _pe_exc:
            logger.debug(f"PromptEvolver error (non-fatal): {_pe_exc}")

        # Build the step prompt
        step_builder = StepPromptBuilder(
            browser_state=browser_state,
            task=self.goal,
            step_number=self.step_number,
            max_steps=self.max_steps,
            agent_history=self._build_history_text(),
            native_ui=native_ui,
            page_snapshot=page_snapshot,
            memories=recalled,
            click_zoom_b64=self._last_click_zoom_b64,
            click_zoom_coords=self._last_click_coords,
            affordances=self._build_affordances(page_snapshot, native_ui),
            evolution_hint=_evolution_hint,
        )
        # Click zoom is consumed once per step — clear after passing to builder
        self._last_click_zoom_b64 = None
        self._last_click_coords = None

        user_messages = step_builder.build_messages(use_vision=_use_vision_this_step)

        # Construct full message list
        messages = [
            {"role": "system", "content": self.system_prompt},
            *user_messages,
        ]

        # Estimate prompt size and warn user on first step (cold prompt processing is slow)
        prompt_tokens_est = sum(len(str(m.get("content", ""))) // 4 for m in messages)
        if self.step_number == 0:
            logger.info(
                f"🧠 Step 1: processing prompt (~{prompt_tokens_est} tokens) — "
                f"first step takes longer as the model loads context. Please wait..."
            )
        else:
            logger.info(f"🧠 Step {self.step_number + 1}: thinking (~{prompt_tokens_est} tokens)...")

        # Retry LLM call up to 3 times with increasing backoff
        agent_output = None
        for _llm_attempt in range(1, 4):
            agent_output = await self._call_llm(messages)
            if agent_output is not None:
                break
            backoff = _llm_attempt * 5  # 5s, 10s, 15s
            logger.warning(
                f"Step {self.step_number + 1}: LLM attempt {_llm_attempt}/3 failed. "
                f"Retrying in {backoff}s..."
            )
            await asyncio.sleep(backoff)

        if agent_output is not None:
            self._consecutive_llm_failures = 0  # Reset circuit breaker on success

        if agent_output is None:
            self._consecutive_llm_failures += 1
            logger.error(
                f"Step {self.step_number + 1}: All 3 LLM attempts failed. "
                f"(Circuit breaker: {self._consecutive_llm_failures}/3 consecutive step failures)"
            )

            # Circuit breaker: if LLM fails 3 steps in a row, the service is likely down
            if self._consecutive_llm_failures >= 3:
                logger.error("🔌 CIRCUIT BREAKER: LLM failed 3 consecutive steps. Pausing for human review.")
                self.pending_auth = {
                    "url": getattr(self.page, "url", ""),
                    "type": "escalate",
                    "message": (
                        "LLM service appears to be down — failed 3 consecutive steps. "
                        "Check API key, model availability, and network connectivity. "
                        "Resume when the issue is resolved."
                    ),
                }
                self._paused = True
                self._save_checkpoint()
            fallback_output = AgentOutput(
                thinking="LLM call failed after 3 retries (API limits, model error, or network issue).",
                next_goal="Retry current step",
                action=[]
            )
            entry = StepHistoryEntry(
                step_number=self.step_number,
                agent_output=fallback_output,
                action_results=[ActionResult(action_name="llm_call", success=False, error="All LLM retry attempts exhausted")],
                url_before=url_before,
                url_after=self.page.url,
            )
            self.history.append(entry)
            return None

        # logger.info(
        #     f"Step {self.step_number + 1}: "
        #     f"Goal: {agent_output.next_goal} | "
        #     f"Actions: {len(agent_output.action)}"
        # )

        # Capture narrative for dashboard display
        if agent_output.narrative:
            self._last_narrative = agent_output.narrative
        elif agent_output.next_goal:
            self._last_narrative = agent_output.next_goal

        # Check pause again — LLM call may have taken 30s so the flag may have
        # been set while we were waiting. This makes pause feel immediate.
        while self._paused:
            await asyncio.sleep(0.3)

        # ─── 3. ACT (hold ScreenLock so only one task touches the computer) ───
        logger.debug(f"Step {self.step_number + 1}: Acting...")
        async with screen_lock.acquire(
            task_id=self._task_id,
            goal=agent_output.next_goal[:80],
        ):
            # Context switch: another task was using the screen since we released it.
            # Bring our window back into focus before acting — same as a human
            # clicking back to their tab after switching to another program.
            if screen_lock.context_switched:
                hint = screen_lock.get_window_hint(self._task_id)
                if hint:
                    try:
                        await asyncio.to_thread(self.computer.display.focus, hint)
                        await asyncio.sleep(0.35)  # let the window fully render
                        logger.info(f"🖥️  Window refocused: '{hint}' (task {self._task_id})")
                    except Exception as _fe:
                        logger.debug(f"Window focus failed: {_fe}")

            action_results = await self._execute_actions(
                agent_output.action,
                browser_state,
            )

        # Record what action types were taken — used by smart vision skip next step
        self._last_action_types = [
            a.computer_call.call.split("(")[0].replace("computer.", "")
            if a.computer_call else a.action_name
            for a in agent_output.action
        ]

        # ─── 4. RECORD + REACTIVE TRACKING ───
        # Read the real current URL from CDP — HumanModeEmulator._url is only
        # updated by goto(), so click-based navigation would leave it stale.
        url_after = self.page.url
        try:
            from autobot.dom.page_snapshot import _get_current_url_sync
            _real_url = await asyncio.to_thread(_get_current_url_sync)
            if _real_url:
                url_after = _real_url
                # Sync back so loop detection and tab list see the correct URL
                if hasattr(self.page, '_url'):
                    self.page._url = _real_url
        except Exception:
            pass
        entry = StepHistoryEntry(
            step_number=self.step_number,
            agent_output=agent_output,
            action_results=action_results,
            url_before=url_before,
            url_after=url_after,
        )
        self.history.append(entry)

        # Track consecutive failures for adaptive retry
        any_failed = any(not r.success for r in action_results)
        any_success = any(r.success for r in action_results)
        current_goal = agent_output.next_goal.strip().lower()[:50]

        # Track click coordinates for drift detection
        for action in agent_output.action:
            if action.computer_call and "mouse.click" in (action.computer_call.call or ""):
                import re as _re
                _m = _re.search(r"x=(\d+).*?y=(\d+)", action.computer_call.call)
                if _m:
                    self._recent_clicks.append((int(_m.group(1)), int(_m.group(2))))
                    if len(self._recent_clicks) > 8:
                        self._recent_clicks = self._recent_clicks[-8:]

        # Detect coordinate drift: clicking same small area repeatedly with slight adjustments
        _click_drift = False
        if len(self._recent_clicks) >= 4:
            _last4 = self._recent_clicks[-4:]
            _xs = [c[0] for c in _last4]
            _ys = [c[1] for c in _last4]
            _x_spread = max(_xs) - min(_xs)
            _y_spread = max(_ys) - min(_ys)
            if _x_spread < 80 and _y_spread < 80:
                _click_drift = True

        if any_failed and current_goal == self._last_goal:
            self._consecutive_failures += 1
            logger.warning(f"Consecutive failure #{self._consecutive_failures} on goal: {current_goal}")
        elif any_failed:
            self._consecutive_failures = 1
        else:
            self._consecutive_failures = 0
            self._recent_clicks.clear()  # reset click tracking on success
        self._last_goal = current_goal

        # Inject failure-specific recovery guidance
        if self._consecutive_failures >= 1:
            # Analyze WHAT failed to give specific advice
            _failed_actions = [
                (a, r) for a, r in zip(agent_output.action, action_results) if not r.success
            ]
            _error_types = set()
            for _a, _r in _failed_actions:
                _err = (_r.error or "").lower()
                if "timeout" in _err or "timed out" in _err:
                    _error_types.add("timeout")
                elif "not found" in _err or "404" in _err:
                    _error_types.add("not_found")
                elif "permission" in _err or "denied" in _err:
                    _error_types.add("permission")
                elif _a.computer_call and "mouse.click" in (_a.computer_call.call or ""):
                    _error_types.add("click_failed")
                else:
                    _error_types.add("other")

            # Build specific recovery hint based on what went wrong
            if _click_drift and self._consecutive_failures >= 2:
                _hint = (
                    "COORDINATE DRIFT DETECTED: You've clicked the same small area 4+ times "
                    "with slight adjustments — the target is likely not where you think. "
                    "STOP clicking. Try: (1) Use a keyboard shortcut instead (Tab, Enter, arrows), "
                    "(2) Check if there's a DOM element [N] you can target precisely, "
                    "(3) Scroll to reveal the real target, or (4) Navigate to a different page."
                )
                self._recent_clicks.clear()
            elif "timeout" in _error_types:
                _hint = "TIMEOUT: The action timed out. The page may be loading slowly. Try wait(5) then retry, or refresh with F5."
            elif "not_found" in _error_types:
                _hint = "NOT FOUND: The target doesn't exist. Check the URL, verify the page loaded, or try a different path."
            elif "permission" in _error_types:
                _hint = "PERMISSION DENIED: You don't have access. Try a different approach or check if you need to log in first."
            elif "click_failed" in _error_types:
                _hint = "CLICK HAD NO EFFECT: The target may be off-screen, disabled, or covered by an overlay. Try scrolling, using keyboard shortcuts, or checking for popups."
            else:
                _FALLBACK_STRATEGIES = [
                    "Try a keyboard shortcut instead of clicking (Tab, Enter, arrows).",
                    "Scroll down/up — the target element may be off-screen.",
                    "Refresh the page (F5) and wait 3 seconds before retrying.",
                    "Try a completely different URL or approach to reach the same destination.",
                ]
                _hint = _FALLBACK_STRATEGIES[self._consecutive_failures % len(_FALLBACK_STRATEGIES)]

            _hint_key = f"[RECOVERY #{self._consecutive_failures}]"
            if not any(_hint_key in s for s in self.scratchpad[-5:]):
                self.scratchpad.append(f"{_hint_key} {_hint}")

            # Auto-remember failed approaches after 2 consecutive failures
            if self._consecutive_failures == 2:
                import hashlib as _hl
                key = "failed_" + _hl.md5(f"{self.goal[:40]}{current_goal[:40]}".encode()).hexdigest()[:8]
                failed_actions = "; ".join(
                    a.computer_call.call[:60] if a.computer_call else a.action_name
                    for a in agent_output.action
                ) or current_goal[:80]
                self._memory_store.remember(key, f"FAILED 2x on '{current_goal[:60]}': {failed_actions}")
                logger.info(f"🧠 Auto-remembered failure: {key}")

            # After 4 consecutive failures, FORCE a strategy change via strong scratchpad alert
            if self._consecutive_failures >= 4:
                self.scratchpad.append(
                    "[FORCED STRATEGY CHANGE] You have failed 4+ times on the same goal. "
                    "Your current approach is NOT working. You MUST try something completely "
                    "different: a different tool, a different page, a different method. "
                    "If you repeat the same action, you will continue to fail."
                )

        # Confidence tracking: low confidence → nudge agent to verify before next action
        confidence = getattr(agent_output, "confidence", "high")
        if confidence == "low" and self._consecutive_failures == 0:
            # Agent is uncertain but hasn't failed yet — warn it to be careful
            if not any("[LOW CONFIDENCE]" in s for s in self.scratchpad[-3:]):
                self.scratchpad.append(
                    "[LOW CONFIDENCE] You reported low confidence. Before your next action: "
                    "(1) Take a screenshot to re-observe the current state, "
                    "(2) Check if there's a DOM element you can target precisely, "
                    "(3) Consider using a more reliable tool (navigate > click, keyboard > mouse)."
                )

        # Watchdog: update last progress time when something meaningful happened
        page_changed = url_after != url_before
        if any_success and (page_changed or not any_failed):
            self._last_progress_time = time.time()

        # Parse special directives from agent memory field
        if agent_output.memory:
            for line in agent_output.memory.splitlines():
                # METRIC:key=value — update numeric metric tracker
                if line.startswith("METRIC:"):
                    try:
                        kv = line[7:].strip()
                        k, v = kv.split("=", 1)
                        self.metrics[k.strip()] = float(v.strip())
                        logger.info(f"📈 Metric update: {k.strip()} = {v.strip()}")
                    except Exception:
                        pass
                # REMEMBER:key=value — persist fact to cross-run memory store
                elif line.startswith("REMEMBER:"):
                    try:
                        kv = line[9:].strip()
                        k, v = kv.split("=", 1)
                        self._memory_store.remember(k.strip(), v.strip())
                    except Exception:
                        pass

        # Accumulate scratchpad from agent's memory (capture key findings)
        if agent_output.memory and len(agent_output.memory) > 20:
            # Suppress only very short/generic filler entries — keep substantive "retry" notes
            # that contain actual context (URLs, values, error messages)
            _mem = agent_output.memory.lower()
            _is_pure_filler = (
                len(agent_output.memory) < 50
                and any(kw in _mem for kw in ("retry", "trying again", "attempt"))
                and not any(useful in _mem for useful in ("http", "error", "fail", "url", "file", "step"))
            )
            if not _is_pure_filler:
                self.scratchpad.append(f"[Step {self.step_number + 1}] {agent_output.memory}")
                # Keep scratchpad manageable
                if len(self.scratchpad) > 20:
                    self.scratchpad = self.scratchpad[-15:]

        # Detect login/auth situations from agent's output
        thinking_lower = agent_output.thinking.lower()
        confidence = getattr(agent_output, 'confidence', 'high')
        auth_keywords = ("login", "log in", "sign in", "signin", "authentication", "password", "credentials")
        if any(kw in thinking_lower for kw in auth_keywords) and confidence in ("low", "medium"):
            self.pending_auth = {
                "url": url_after,
                "type": "login_detected",
                "message": f"Login page detected at {url_after}. Agent confidence: {confidence}. "
                           f"Agent says: {agent_output.thinking[:200]}",
            }
            logger.info(f"🔐 Authentication detected at {url_after}")

        step_time = time.time() - step_start
        logger.debug(f"Step {self.step_number + 1} completed in {step_time:.1f}s")

        # ─── RL: Record experience for every action taken this step ───
        try:
            if self._rl is not None and agent_output.action:
                for action, result in zip(agent_output.action, action_results):
                    _params: dict = {}
                    if action.computer_call:
                        _params = {"call": action.computer_call.call}
                    elif action.navigate:
                        _params = {"url": action.navigate.url}
                    elif action.click:
                        _params = {"index": action.click.index}
                    elif action.input_text:
                        _params = {"index": action.input_text.index}

                    _task_done = action.done is not None
                    _task_success = (action.done.success if action.done else None)

                    self._rl.record_step(
                        url=url_before,
                        goal=self.goal,
                        action_name=action.action_name,
                        action_params=_params,
                        success=result.success,
                        error=result.error,
                        step_number=self.step_number,
                        url_before=url_before,
                        url_after=url_after,
                        current_goal=agent_output.next_goal,
                        consecutive_failures=self._consecutive_failures,
                        same_goal_count=sum(
                            1 for e in self.history[-8:]
                            if e.agent_output.next_goal.strip().lower()[:50] == current_goal
                        ),
                        coordinate_drift=_click_drift,
                        llm_circuit_breaker=(self._consecutive_llm_failures >= 3),
                        task_done=_task_done,
                        task_success=_task_success,
                    )
        except Exception as _rl_exc:
            logger.debug(f"RL record_step failed (non-fatal): {_rl_exc}")

        # Check if agent called "done"
        for action in agent_output.action:
            if action.done is not None:
                return action.done.text

        return None

    async def _call_llm(self, messages: list[dict]) -> AgentOutput | None:
        """
        Call the LLM with the current state and parse the structured output.
        _make_llm_call handles format/vision fallbacks internally.
        This layer adds one outer retry for transient network errors.
        """
        for attempt in range(1, 3):
            try:
                _t0 = time.time()
                response = await self._make_llm_call(messages)
                _elapsed = time.time() - _t0
                logger.info(f"⏱️ LLM call took {_elapsed:.1f}s (step {self.step_number + 1}, attempt {attempt})")
                if not response:
                    logger.warning(f"LLM outer attempt {attempt}: empty string returned")
                    await asyncio.sleep(5)
                    continue
                result = self._parse_agent_output(response)
                if result is not None:
                    return result
                logger.warning(f"LLM outer attempt {attempt}: could not parse JSON. Raw: {response[:300]}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"LLM outer attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt < 2:
                    await asyncio.sleep(10)

        logger.error(f"All LLM attempts failed for model '{self.model}'. "
                     f"Check OPENROUTER_API_KEY and model availability.")
        return None

    async def _make_llm_call(self, messages: list[dict]) -> str:
        """
        Make the actual LLM API call using the async client.

        Attempt order:
          1. Full request — vision + JSON mode (best quality)
          2. No JSON mode — for models that don't support response_format
          3. Text-only — strip screenshots for models that reject large image payloads

        Free OpenRouter models frequently return empty responses when sent large
        base64 images, so the text-only fallback is critical for reliability.
        """
        # Multi-LLM routing: use fast_model for routine steps (step > 0 and not a REPLAN).
        # Step 0 is planning (needs full intelligence); REPLAN also uses primary model.
        is_planning_step = self.step_number == 0
        is_replan = self._last_eval_signal == "replan"
        active_model = (
            self.model
            if (not self.fast_model or is_planning_step or is_replan)
            else self.fast_model
        )
        if self.fast_model and active_model == self.fast_model:
            logger.debug(f"[multi-LLM] Step {self.step_number}: using fast model ({self.fast_model})")

        args = {
            "model": active_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,   # Agent needs room for detailed thinking + coordinate reasoning
            "response_format": {"type": "json_object"},
        }

        # Output token budget — only tighten for budget/fast models.
        # Premium models (Grok, Claude Opus, GPT-4o, Gemini Pro) run uncapped so their
        # full reasoning capacity is preserved. Only "flash" and "lite" variants are tightened
        # since they're designed for speed and have lower default output limits anyway.
        _is_budget_model = any(
            kw in active_model.lower() for kw in ("flash", "lite", "mini", "haiku")
        )
        if _is_budget_model and not is_planning_step and not is_replan:
            args["max_tokens"] = int(os.getenv("AUTOBOT_MAX_TOKENS_ROUTINE", "2048"))

        # Gemini Flash/Lite only: cap the internal chain-of-thought (thinkingBudget).
        # Never applied to Grok, Claude, GPT-4o, Gemini Pro, or any full-power model.
        # Set AUTOBOT_THINKING_BUDGET=-1 to disable even on flash models (full reasoning).
        _thinking_budget = int(os.getenv("AUTOBOT_THINKING_BUDGET", "1024"))
        if (
            _thinking_budget >= 0
            and _is_budget_model
            and "gemini" in active_model.lower()
            and not is_planning_step
        ):
            args.setdefault("extra_body", {}).update({
                "generationConfig": {
                    "thinkingConfig": {"thinkingBudget": _thinking_budget}
                }
            })

        # Gemini: route system prompt via system_instruction for implicit prefix caching.
        # Gemini caches identical system_instruction content across calls at ~10% of input token cost.
        # Env: AUTOBOT_PROMPT_CACHE=0 disables this (use for non-Gemini compatible endpoints).
        if os.getenv("AUTOBOT_PROMPT_CACHE", "1") == "1":
            _base_url = str(getattr(self.llm_client, "base_url", "") or "")
            if "generativelanguage.googleapis.com" in _base_url:
                _sys = [m for m in args["messages"] if m["role"] == "system"]
                _usr = [m for m in args["messages"] if m["role"] != "system"]
                if _sys:
                    args.setdefault("extra_body", {})["system_instruction"] = {
                        "parts": [{"text": _sys[0]["content"]}]
                    }
                    args["messages"] = _usr
                    logger.debug("Gemini: system prompt routed via system_instruction (prefix caching enabled)")

        async def _do_call(current_args: dict) -> str:
            # Wrap in a Task so cancel() can interrupt in-flight API calls
            async def _inner():
                resp = await self.llm_client.chat.completions.create(**current_args)
                if not resp.choices:
                    raise ValueError(f"No choices returned by {current_args['model']}")
                content = resp.choices[0].message.content
                if not content or not content.strip():
                    finish = getattr(resp.choices[0], "finish_reason", "unknown")
                    raise ValueError(f"Empty content from {current_args['model']} (finish_reason={finish})")
                return str(content)

            task = asyncio.create_task(_inner())
            self._current_llm_task = task
            try:
                return await task
            except asyncio.CancelledError:
                logger.info("🛑 LLM call cancelled")
                raise
            finally:
                self._current_llm_task = None

        last_error: Exception | None = None

        # Fast-path: if this model is already known to not support vision, skip
        # straight to text-only. Avoids 2 wasted 404 round-trips per step.
        # A model is flagged after its first "No endpoints found that support image input"
        # error so that all subsequent steps go text-only immediately.
        _no_vision_key = f"_no_vision_{active_model}"
        _model_no_vision = getattr(self, _no_vision_key, False)

        # Also skip vision for models we know can't handle images
        _TEXT_ONLY_MODELS = (
            "deepseek", "llama", "mistral", "codestral", "qwen",
            "o1-mini", "o1-preview", "command-r",
        )
        if not _model_no_vision and any(kw in active_model.lower() for kw in _TEXT_ONLY_MODELS):
            _model_no_vision = True
            setattr(self, _no_vision_key, True)
            logger.info(f"Model '{active_model}' flagged as text-only — skipping vision attempts")

        if _model_no_vision:
            # Go straight to text-only, no wasted attempts
            text_only_msgs = _strip_images_from_messages(messages)
            text_only_args = {**args, "messages": text_only_msgs}
            text_only_args.pop("response_format", None)
            try:
                return await _do_call(text_only_args)
            except Exception as e:
                raise e

        # Attempt 1: vision + JSON mode
        try:
            return await _do_call(args)
        except Exception as e:
            last_error = e
            logger.warning(f"LLM attempt 1 ({self.model}): {e}")
            # If the model says it doesn't support image input, flag it for future steps
            if "image input" in str(e).lower() or "no endpoints found" in str(e).lower():
                setattr(self, _no_vision_key, True)
                logger.info(f"Model '{active_model}' auto-flagged as text-only after vision rejection")

        # Attempt 2: drop response_format.
        # OpenRouter and many models return 400 / empty / "unsupported" when JSON mode is
        # requested for a model that doesn't support it. Broad trigger: retry without it
        # unless the error is clearly about billing/auth (401/403/429).
        err_lower = str(last_error).lower()

        # 429 / rate limit = transient — retry with backoff, don't skip fallback attempts
        _is_rate_limited = "429" in err_lower or "rate limit" in err_lower or "resource_exhausted" in err_lower
        if _is_rate_limited:
            logger.warning("Rate limited — waiting 10s before retry...")
            await asyncio.sleep(10)

        _hard_errors = ("401", "403", "insufficient_quota")
        # Also skip attempt 2 if we just learned the model doesn't support vision —
        # go straight to attempt 3 (text-only) to avoid another wasted round-trip.
        _just_flagged_no_vision = "image input" in err_lower or "no endpoints found" in err_lower
        if not any(kw in err_lower for kw in _hard_errors) and not _just_flagged_no_vision:
            try:
                no_format = {**args}
                no_format.pop("response_format", None)
                logger.info(f"Retrying {self.model} without JSON mode...")
                return await _do_call(no_format)
            except Exception as e2:
                last_error = e2
                logger.warning(f"LLM attempt 2 (no JSON mode): {e2}")

        # Attempt 3: strip images — model rejected large payload or doesn't support vision
        text_only = _strip_images_from_messages(messages)
        if text_only != messages:
            try:
                no_vision = {**args, "messages": text_only}
                no_vision.pop("response_format", None)
                logger.info(f"Retrying {self.model} without images (text-only fallback)...")
                return await _do_call(no_vision)
            except Exception as e3:
                last_error = e3
                logger.warning(f"LLM attempt 3 (text-only): {e3}")

        raise last_error

    @staticmethod
    def _normalize_agent_data(data: dict) -> dict:
        """
        Normalise raw LLM JSON before Pydantic validation.

        Handles common LLM deviations:
        - action is null / missing / a dict instead of a list
        - action items are bare strings like "computer.mouse.click(x=5,y=10)"
        - action items use {"call": "..."} instead of {"computer_call": {"call": "..."}}
        - next_goal or thinking are None instead of empty string
        """
        # Ensure action is always a list
        raw_action = data.get("action")
        if raw_action is None:
            data["action"] = []
        elif isinstance(raw_action, dict):
            data["action"] = [raw_action]
        elif isinstance(raw_action, str):
            # Bare string action — wrap as computer_call if it looks like one
            if raw_action.startswith("computer."):
                data["action"] = [{"computer_call": {"call": raw_action}}]
            else:
                data["action"] = []
        elif isinstance(raw_action, list):
            normalised = []
            for item in raw_action:
                if isinstance(item, str):
                    if item.startswith("computer."):
                        normalised.append({"computer_call": {"call": item}})
                    # else skip non-parseable strings
                elif isinstance(item, dict):
                    # {"call": "computer.mouse.click(...)"} → {"computer_call": {"call": ...}}
                    if "call" in item and "computer_call" not in item:
                        normalised.append({"computer_call": {"call": item["call"]}})
                    else:
                        normalised.append(item)
            data["action"] = normalised

        # Coerce None strings to empty
        for key in ("thinking", "next_goal", "memory", "narrative",
                    "evaluation_previous_goal", "confidence"):
            if data.get(key) is None:
                data[key] = ""

        return data

    def _parse_agent_output(self, raw: str) -> AgentOutput | None:
        """Parse the LLM's JSON response into an AgentOutput model."""
        from pydantic import ValidationError
        import re as _re_parse

        def _try_build(data: dict) -> AgentOutput | None:
            try:
                result = AgentOutput(**self._normalize_agent_data(data))
                # Log if all actions came back unknown — model used wrong format
                if result.action and all(a.action_name == "unknown" for a in result.action):
                    logger.warning(f"Model output parsed but all actions are unknown. "
                                   f"Raw action data: {data.get('action')}")
                return result
            except (ValidationError, TypeError) as e:
                logger.debug(f"AgentOutput build failed: {e}")
                return None

        try:
            text = raw.strip()

            # 1. Direct JSON parse
            try:
                data = json.loads(text)
                result = _try_build(data)
                if result is not None:
                    return result
            except json.JSONDecodeError:
                pass

            # 2. Extract from markdown code block
            json_match = _re_parse.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re_parse.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    result = _try_build(data)
                    if result is not None:
                        return result
                except json.JSONDecodeError:
                    pass

            # 3. Bracket-matching outermost { ... }
            json_str = self._extract_outermost_json(text)
            if json_str:
                try:
                    data = json.loads(json_str)
                    result = _try_build(data)
                    if result is not None:
                        return result
                except json.JSONDecodeError:
                    pass

            # 4. Last resort: if the model returned plain text with a computer_call embedded,
            #    extract it and wrap into a minimal AgentOutput so the agent doesn't deadlock.
            cc_match = _re_parse.search(r'computer\.\w+\.\w+\([^)]*\)', text)
            if cc_match:
                logger.warning("LLM returned non-JSON — extracting embedded computer_call as fallback")
                return AgentOutput(
                    thinking=text[:200],
                    next_goal="Execute extracted action",
                    action=[{"computer_call": {"call": cc_match.group(0)}}],
                )

            logger.error(f"All JSON parse attempts failed. Raw[:500]: {raw[:500]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during agent output parsing: {e}")
            return None

    @staticmethod
    def _extract_outermost_json(text: str) -> str | None:
        """Extract the outermost JSON object from text using bracket matching."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    async def _execute_actions(
        self,
        actions: list[ActionModel],
        browser_state: BrowserState,
    ) -> list[ActionResult]:
        """
        Execute a list of actions sequentially.

        Adapted from Browser Use: if an action changes the page
        (navigation, click on link), remaining actions are SKIPPED.
        """
        results: list[ActionResult] = []

        for i, action in enumerate(actions):
            # Check if this is a "done" action — don't actually execute, just record
            if action.done is not None:
                results.append(ActionResult(
                    action_name="done",
                    success=True,
                    extracted_content=action.done.text,
                ))
                break

            # ── Approval gate ────────────────────────────────────────────────
            tier = self._approval_guard.classify(action)
            if tier != RiskTier.SAFE:
                allowed = await self._approval_guard.gate(
                    action=action,
                    tier=tier,
                    goal=getattr(self, '_last_goal', ''),
                )
                if not allowed:
                    results.append(ActionResult(
                        action_name=action.action_name,
                        success=False,
                        error=f"Blocked by approval guard ({tier.value}): user denied or timed out",
                    ))
                    continue

            # OS dialog detection: snapshot window titles before actions that could spawn
            # a system dialog (file picker, auth prompt, GTK dialog, Electron modal).
            # These are invisible to DOM-based popup detection.
            _dialog_detect = os.getenv("AUTOBOT_DIALOG_DETECT", "1") == "1"
            _might_spawn_dialog = (
                action.computer_call is not None or action.navigate is not None
            )
            _wins_before: frozenset = frozenset()
            if _dialog_detect and _might_spawn_dialog:
                _wins_before = await asyncio.to_thread(
                    self.computer.display.window_titles
                )

            # Execute the action
            url_before = self.page.url
            result = await self._execute_action(action, browser_state)
            results.append(result)

            # Check for new OS windows after the action
            if _dialog_detect and _might_spawn_dialog and _wins_before:
                _wins_after = await asyncio.to_thread(
                    self.computer.display.window_titles
                )
                for _wt in (_wins_after - _wins_before):
                    # Skip new Chrome/browser tabs — those are expected
                    if any(skip in _wt.lower() for skip in ("chrome", "chromium", "firefox", "brave")):
                        continue
                    _alert = (
                        f"[SYSTEM DIALOG: '{_wt}'] A new OS window appeared after the last action. "
                        "Take a screenshot — it may be a file picker, permission prompt, or save dialog. "
                        "Handle it before continuing."
                    )
                    if not any(_wt in s for s in self.scratchpad[-3:]):
                        self.scratchpad.append(_alert)
                        logger.info(f"OS dialog detected: '{_wt}'")

            # Page change detection (adapted from Browser Use)
            # If the page changed, skip remaining actions
            if result.page_changed or self.page.url != url_before:
                remaining = len(actions) - i - 1
                if remaining > 0:
                    logger.info(
                        f"Page changed after action {i + 1}, "
                        f"skipping {remaining} remaining actions"
                    )
                break

            # Small delay between actions to mimic human behavior
            await asyncio.sleep(0.3)

        return results

    async def _execute_action(
        self,
        action: ActionModel,
        browser_state: BrowserState,
    ) -> ActionResult:
        """
        Execute a single action on the browser or OS.

        Key innovation from Browser Use: click/input use DOM INDEX, not selectors.
        """
        action_name = action.action_name

        try:
            if action.navigate is not None:
                target_url = action.navigate.url
                # Any navigate means we're in the browser — update window hint
                self._window_hint = "Google Chrome"
                screen_lock.register_window(self._task_id, "Google Chrome")
                await self.page.goto(target_url)
                # Detect obvious page-load errors (redirect to error page, offline, etc.)
                landed_url = self.page.url
                _ERROR_SIGNALS = ("404", "403", "500", "503", "error", "not-found",
                                  "page-not-found", "unavailable", "offline", "chrome-error")
                url_lower = landed_url.lower()
                if any(sig in url_lower for sig in _ERROR_SIGNALS):
                    self.scratchpad.append(
                        f"[PAGE ERROR] Navigated to '{target_url}' but landed on '{landed_url}' — "
                        "possible 404/error page. Try a different URL or check if the site is down."
                    )
                    logger.warning(f"⚠️ Page error detected after navigate: {landed_url}")
                    return ActionResult(
                        action_name="navigate",
                        success=False,
                        error=f"Page load error — landed on '{landed_url}'. The page may be down or the URL wrong.",
                        page_changed=True,
                    )
                return ActionResult(action_name="navigate", success=True, page_changed=True)

            elif action.click is not None:
                return ActionResult(
                    action_name="click", 
                    success=False, 
                    error="DOM Click not available. Use computer.mouse.click(x, y) instead based on visual coordinates in the screenshot."
                )

            elif action.input_text is not None:
                return ActionResult(
                    action_name="input_text", 
                    success=False, 
                    error="DOM Input not available. Use computer.mouse.click() to focus then computer.keyboard.type() instead."
                )

            elif action.scroll_down is not None:
                await self.page.evaluate(
                    f"window.scrollBy(0, {action.scroll_down.amount * 300})"
                )
                return ActionResult(action_name="scroll_down", success=True)

            elif action.scroll_up is not None:
                await self.page.evaluate(
                    f"window.scrollBy(0, -{action.scroll_up.amount * 300})"
                )
                return ActionResult(action_name="scroll_up", success=True)

            elif action.press_key is not None:
                await self.page.keyboard.press(action.press_key.key)
                return ActionResult(action_name="press_key", success=True)

            elif action.go_back is not None:
                await self.page.go_back()
                return ActionResult(action_name="go_back", success=True, page_changed=True)

            elif action.new_tab is not None:
                new_page = await self.page.context.new_page()
                if action.new_tab.url != "about:blank":
                    await new_page.goto(action.new_tab.url, wait_until="domcontentloaded")
                self.page = new_page  # Switch focus to new tab
                return ActionResult(action_name="new_tab", success=True, page_changed=True)

            elif action.switch_tab is not None:
                tab_id = action.switch_tab.tab_id
                for p in self.page.context.pages:
                    if str(getattr(p, 'tab_index', '')) == tab_id:
                        self.page = p
                        await p.bring_to_front()
                        return ActionResult(action_name="switch_tab", success=True, page_changed=True)
                available = [str(getattr(p, 'tab_index', '?')) for p in self.page.context.pages]
                return ActionResult(
                    action_name="switch_tab",
                    success=False,
                    error=f"Tab {tab_id} not found. Available tabs: {available}",
                )

            elif action.close_tab is not None:
                await self.page.close()
                pages = self.page.context.pages
                if pages:
                    self.page = pages[-1]
                return ActionResult(action_name="close_tab", success=True, page_changed=True)

            elif action.wait is not None:
                actual = await self._smart_wait(hint_seconds=action.wait.seconds)
                return ActionResult(
                    action_name="wait",
                    success=True,
                    extracted_content=f"Waited {actual:.1f}s (adaptive)",
                )

            elif action.screenshot is not None:
                return ActionResult(action_name="screenshot", success=True)

            elif action.computer_call is not None:
                return await self._execute_computer_call(action.computer_call)

            elif action.click_native is not None:
                success = self.computer.window.click(action.click_native.index)
                return ActionResult(action_name="click_native", success=success)

            elif action.input_text_native is not None:
                success = self.computer.window.type(action.input_text_native.index, action.input_text_native.text)
                return ActionResult(action_name="input_text_native", success=success)

            else:
                return ActionResult(
                    action_name="unknown",
                    success=False,
                    error=f"Unknown action: {action_name}",
                )

        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            return ActionResult(action_name=action_name, success=False, error=str(e))

    async def _execute_click(self, click: ClickAction, browser_state: BrowserState) -> ActionResult:
        """
        Click an element by its DOM index.

        This is the key pattern from Browser Use: instead of guessing CSS selectors,
        the LLM references elements by their numeric index from the DOM tree.
        We resolve the index to an actual element using the accessibility tree.
        """
        index = click.index
        element = browser_state.selector_map.get(index)

        if element is None:
            return ActionResult(
                action_name="click",
                success=False,
                error=f"Element with index {index} not found in selector map",
            )

        try:
            # Strategy: Use accessibility role + name to find the element via Playwright
            role = element.attributes.get("role", "")
            text = element.text

            if role and text:
                await self.page.get_by_role(role, name=text).first.click(timeout=5000)
            elif text:
                await self.page.get_by_text(text, exact=False).first.click(timeout=5000)
            elif element.tag_name == "a" and "href" in element.attributes:
                await self.page.locator(f'a[href="{element.attributes["href"]}"]').first.click(timeout=5000)
            else:
                # Fallback: use tag + any identifying attribute
                selector = element.tag_name
                if "name" in element.attributes:
                    selector = f'{element.tag_name}[name="{element.attributes["name"]}"]'
                elif "aria-label" in element.attributes:
                    selector = f'{element.tag_name}[aria-label="{element.attributes["aria-label"]}"]'

                await self.page.locator(selector).first.click(timeout=5000)

            logger.info(f"Clicked [{index}] <{element.tag_name}> '{text[:30]}'")

            # Check if page changed
            await asyncio.sleep(0.5)
            page_changed = self.page.url != browser_state.url

            return ActionResult(action_name="click", success=True, page_changed=page_changed)

        except Exception as e:
            return ActionResult(
                action_name="click",
                success=False,
                error=f"Click on [{index}] failed: {e}",
            )

    async def _execute_input(self, input_action: InputTextAction, browser_state: BrowserState) -> ActionResult:
        """
        Type text into an element by its DOM index.
        Same index-based approach as click.
        """
        index = input_action.index
        element = browser_state.selector_map.get(index)

        if element is None:
            return ActionResult(
                action_name="input_text",
                success=False,
                error=f"Element with index {index} not found in selector map",
            )

        try:
            role = element.attributes.get("role", "")
            text = element.text
            placeholder = element.attributes.get("placeholder", "")

            if role in ("textbox", "searchbox", "combobox"):
                locator = self.page.get_by_role(role, name=text or placeholder)
            elif placeholder:
                locator = self.page.get_by_placeholder(placeholder)
            elif text:
                locator = self.page.get_by_label(text)
            else:
                locator = self.page.locator(f'{element.tag_name}[name="{element.attributes.get("name", "")}"]')

            await locator.first.click(timeout=5000)
            await locator.first.fill(input_action.text, timeout=5000)

            logger.info(f"Input [{index}] <{element.tag_name}>: '{input_action.text[:30]}'")
            return ActionResult(action_name="input_text", success=True)

        except Exception as e:
            return ActionResult(
                action_name="input_text",
                success=False,
                error=f"Input to [{index}] failed: {e}",
            )

    @staticmethod
    def _compress_screenshot(png_bytes: bytes) -> str:
        """
        Compress a screenshot from PNG to JPEG to reduce token usage.

        IMPORTANT: No resolution change! The LLM sees the image at the SAME
        resolution as the real screen, so coordinates match 1:1. This avoids
        the problem where the LLM estimates coordinates for a downscaled image
        but the mouse clicks at real-screen coordinates.

        JPEG quality is configurable via AUTOBOT_JPEG_QUALITY (default 50).
        Lower quality = fewer tokens = faster LLM calls. 50 is still very
        readable for LLMs while being ~60% smaller than quality 80.
        """
        import base64
        try:
            from PIL import Image
            from io import BytesIO

            _quality = int(os.getenv("AUTOBOT_JPEG_QUALITY", "50"))
            img = Image.open(BytesIO(png_bytes))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=_quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            logger.debug(f"Screenshot compressed: {len(png_bytes)//1024}KB PNG → "
                         f"{len(buf.getvalue())//1024}KB JPEG ({img.size[0]}x{img.size[1]})")
            return b64
        except ImportError:
            logger.debug("Pillow not installed — sending full-size PNG to LLM")
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception as e:
            logger.warning(f"Screenshot compress failed: {e} — using original PNG")
            return base64.b64encode(png_bytes).decode("utf-8")

    async def _smart_wait(self, hint_seconds: float = 5.0) -> float:
        """
        Evidence-based adaptive wait — exits as soon as the process completes,
        never sleeps for a fixed duration.

        The agent calls wait(seconds=N) as a *hint* about how long it thinks
        something might take.  This method treats that hint as guidance, not
        a command, and exits the moment it observes completion:

          - Screen stable:  2 consecutive identical MD5 hashes  → done
          - URL changed:    page navigated to new location      → done (load finished)
          - DOM stable:     element count unchanged for 2 polls  → done (content settled)

        The effective max timeout is derived from:
          1. Learned 90th-percentile for this URL from WaitDurationMemory (most accurate)
          2. hint_seconds × 2  (generous fallback when no history yet)
          3. Hard floor of 5s and hard ceiling of 180s

        After exiting, records the actual duration so future waits for this URL
        start with a better-calibrated max.

        Returns actual seconds elapsed.
        """
        import hashlib
        import re as _re

        # ── Determine effective max ──────────────────────────────────────────
        try:
            from autobot.learning.policy_memory import wait_duration_memory
            current_url = self.page.url if self.page else ""
            # Extract domain+path prefix as URL pattern key
            url_clean = _re.sub(r"^https?://", "", current_url).lower()
            parts = url_clean.split("/")
            domain = parts[0].replace("www.", "")
            url_pattern = f"{domain}/{parts[1][:20]}" if len(parts) > 1 and parts[1] else domain[:40]
            learned_max = wait_duration_memory.get_learned_max(url_pattern)
        except Exception:
            wait_duration_memory = None
            url_pattern = ""
            learned_max = None

        if learned_max is not None:
            # Use learned data: add 50% buffer over the 90th percentile
            effective_max = max(learned_max * 1.5, hint_seconds, 5.0)
            logger.info(
                f"⏳ Smart wait: learned p90={learned_max:.1f}s for {url_pattern!r}, "
                f"max={effective_max:.1f}s (hint={hint_seconds:.1f}s)"
            )
        else:
            # No history yet — use hint × 2 as generous max
            effective_max = max(hint_seconds * 2.0, 5.0)
            logger.info(
                f"⏳ Smart wait: no history for {url_pattern!r}, "
                f"max={effective_max:.1f}s (hint×2, hint={hint_seconds:.1f}s)"
            )
        # Hard ceiling: never block longer than 3 minutes
        effective_max = min(effective_max, 180.0)

        # ── Poll for completion signals ──────────────────────────────────────
        prev_hash: str | None = None
        stable_count = 0
        prev_dom_count: int | None = None
        dom_stable_count = 0
        start_url = self.page.url if self.page else ""
        start = time.time()
        elapsed = 0.0
        check_interval = 1.0   # 1s poll — fast enough to catch quick loads

        while elapsed < effective_max:
            await asyncio.sleep(check_interval)
            elapsed = time.time() - start

            try:
                current_url_now = self.page.url if self.page else ""

                # ── Signal 1: URL changed → navigation completed ──────────
                if current_url_now != start_url:
                    logger.info(
                        f"✅ Smart wait: URL changed after {elapsed:.1f}s — navigation done"
                    )
                    break

                # ── Signal 2: Screenshot hash stable ─────────────────────
                shot = await self.page.screenshot()
                curr_hash = hashlib.md5(shot).hexdigest()

                if curr_hash == prev_hash:
                    stable_count += 1
                    if stable_count >= 2:
                        logger.info(
                            f"✅ Smart wait: screen stable after {elapsed:.1f}s"
                        )
                        break
                else:
                    stable_count = 0
                prev_hash = curr_hash

                # ── Signal 3: DOM element count stable ───────────────────
                try:
                    dom_count = await self.page.evaluate("document.querySelectorAll('*').length")
                    if prev_dom_count is not None and dom_count == prev_dom_count:
                        dom_stable_count += 1
                        if dom_stable_count >= 2 and elapsed >= 1.5:
                            logger.info(
                                f"✅ Smart wait: DOM stable ({dom_count} elements) after {elapsed:.1f}s"
                            )
                            break
                    else:
                        dom_stable_count = 0
                    prev_dom_count = dom_count
                except Exception:
                    pass  # DOM eval unavailable — rely on screen hash

            except Exception as e:
                logger.debug(f"Smart wait poll error: {e}")
                break

        else:
            logger.info(f"⏳ Smart wait: max timeout {effective_max:.0f}s reached — proceeding")

        actual_seconds = time.time() - start

        # ── Record actual duration for future calibration ────────────────────
        try:
            if wait_duration_memory is not None and url_pattern:
                wait_duration_memory.record(url_pattern, actual_seconds)
                wait_duration_memory.save()
        except Exception:
            pass

        return actual_seconds

    async def _execute_computer_call(self, call_action: ComputerCallAction) -> ActionResult:
        """
        Execute an OS-level computer tool call safely via AST parsing.
        Example: "computer.mouse.click(x=640, y=480)"
        """
        call_str = call_action.call
        logger.info(f"💻 Computer call: {call_str}")

        try:
            result = await asyncio.to_thread(self._dispatch_computer_call, call_str)
            await asyncio.sleep(0.8)

            action_result = ActionResult(
                action_name="computer_call",
                success=True,
                extracted_content=str(result) if result is not None else None,
            )

            # Track window context for smooth ScreenLock handoffs.
            # When the agent explicitly focuses a window, update our hint so that
            # after a context switch, we refocus the right window on next lock acquire.
            import re as _re_wh
            _focus_m = _re_wh.search(r"display\.focus\(['\"](.+?)['\"]\)", call_str)
            if _focus_m:
                _wh = _focus_m.group(1)
                self._window_hint = _wh
                screen_lock.register_window(self._task_id, _wh)
                logger.debug(f"Window hint updated: '{_wh}'")
            elif "mouse.click" in call_str or "keyboard.type" in call_str:
                # Most mouse/keyboard actions are in whichever window is currently
                # focused — keep the hint as-is (don't reset it here)
                pass

            # Click zoom: capture a 300×300 region around the click target so the
            # LLM can verify in the NEXT step whether the click landed correctly.
            # Only ~400 tokens vs 10,000–20,000 for a full screenshot retry.
            if "mouse.click" in call_str and os.getenv("AUTOBOT_CLICK_ZOOM", "1") == "1":
                import re as _re
                _m = (
                    _re.search(r"x\s*=\s*(-?\d+).*?y\s*=\s*(-?\d+)", call_str)
                    or _re.search(r"(-?\d+)\s*,\s*(-?\d+)", call_str)
                )
                if _m:
                    _cx, _cy = int(_m.group(1)), int(_m.group(2))
                    try:
                        _sw, _sh = self.computer.display.size()
                        _rx = max(0, _cx - 150)
                        _ry = max(0, _cy - 150)
                        _rw = min(300, _sw - _rx)
                        _rh = min(300, _sh - _ry)
                        import base64 as _b64
                        _zoom_png_b64 = self.computer.display.screenshot_region(_rx, _ry, _rw, _rh)
                        self._last_click_zoom_b64 = self._compress_screenshot(
                            _b64.b64decode(_zoom_png_b64)
                        )
                        self._last_click_coords = (_cx, _cy)
                        logger.debug(f"Click zoom captured at ({_cx},{_cy}) → {_rw}×{_rh}px region")
                    except Exception as _ze:
                        logger.debug(f"Click zoom capture failed: {_ze}")

            return action_result
        except Exception as e:
            logger.error(f"Computer call failed [{call_str}]: {e}")
            return ActionResult(
                action_name="computer_call",
                success=False,
                error=f"{e}. Check the call format: computer.<module>.<method>(<args>)",
            )

    def _dispatch_computer_call(self, call_str: str) -> Any:
        """
        Safely parse and dispatch a computer.* method call without eval().

        Supported format:  computer.<module>.<method>(<args>)
        Examples:
          computer.mouse.click(x=640, y=480)
          computer.keyboard.type('hello world')
          computer.mouse.scroll(0, -5)
          computer.clipboard.set('some text')
        """
        import ast
        import re

        call_str = call_str.strip()
        if not call_str.startswith("computer."):
            raise ValueError(f"Call must start with 'computer.': {call_str}")

        # Match: computer.<module>.<method>(<args>)
        pattern = r"^computer\.(\w+)\.(\w+)\(([^)]*)\)$"
        match = re.match(pattern, call_str)
        if not match:
            # Try with multi-char args that may contain parens (e.g. strings with parens)
            pattern2 = r"^computer\.(\w+)\.(\w+)\((.*)\)$"
            match = re.match(pattern2, call_str, re.DOTALL)
        if not match:
            raise ValueError(f"Cannot parse call: '{call_str}'. "
                             f"Expected: computer.<module>.<method>(<args>)")

        module_name, method_name, args_str = match.groups()

        module = getattr(self.computer, module_name, None)
        if module is None:
            available = [a for a in dir(self.computer) if not a.startswith("_")]
            raise ValueError(f"Unknown computer module '{module_name}'. "
                             f"Available: {', '.join(available)}")

        method = getattr(module, method_name, None)
        if method is None:
            raise ValueError(f"Unknown method 'computer.{module_name}.{method_name}'")

        # Parse arguments using AST (safe, no eval)
        args: list = []
        kwargs: dict = {}
        if args_str.strip():
            try:
                tree = ast.parse(f"_f({args_str})", mode="eval")
                call_node = tree.body
                for arg_node in call_node.args:
                    args.append(ast.literal_eval(arg_node))
                for kw_node in call_node.keywords:
                    kwargs[kw_node.arg] = ast.literal_eval(kw_node.value)
            except Exception as parse_err:
                raise ValueError(f"Cannot parse args '{args_str}': {parse_err}")

        return method(*args, **kwargs)

    def _build_affordances(self, page_snapshot: Any, native_ui: Any) -> str:
        """
        Build a concise affordance summary for the current step.

        Tells the LLM exactly what tools are practically usable RIGHT NOW:
        - Is a DOM available (interactive elements count)?
        - What windows are open (active app context)?
        - What's in the clipboard (can it be pasted immediately)?
        - Are any background terminal processes running?

        This replaces guess-work with factual grounding, reducing failed actions.
        ~100-150 tokens per step, but prevents multiple retry steps.
        """
        lines: list[str] = []

        # DOM status
        if page_snapshot and page_snapshot.num_interactive > 0:
            lines.append(f"DOM: {page_snapshot.num_interactive} interactive elements (see DOM snapshot below)")
        elif page_snapshot and page_snapshot.url:
            lines.append("DOM: browser page loaded but no interactive elements detected (SPA may still be rendering)")
        else:
            lines.append("DOM: unavailable — use mouse/keyboard from screenshot coordinates only")

        # Active windows via wmctrl
        try:
            titles = list(self.computer.display.window_titles())[:5]
            if titles:
                lines.append(f"Open windows: {', '.join(titles)}")
        except Exception:
            pass

        # Clipboard preview
        try:
            clip = self.computer.clipboard.get()
            if clip and clip.strip():
                preview = clip.strip()[:60].replace("\n", " ")
                suffix = "..." if len(clip) > 60 else ""
                lines.append(f"Clipboard: '{preview}{suffix}' ({len(clip)} chars — ready to paste)")
            else:
                lines.append("Clipboard: empty")
        except Exception:
            pass

        # Background terminal processes
        try:
            _procs = getattr(self.computer.terminal, '_procs', {})
            running = [pid for pid in _procs if self.computer.terminal.running(pid)]
            if running:
                lines.append(f"Background processes: {len(running)} running — PIDs {running} (use terminal.output(pid) to check)")
        except Exception:
            pass

        # Page type hint — tells agent which tools work best on this type of page
        try:
            if page_snapshot:
                _page_type_hint = _infer_page_type_hint(page_snapshot)
                if _page_type_hint:
                    lines.append(_page_type_hint)
        except Exception:
            pass

        # RL-learned tool preferences for this page context
        try:
            if self._rl is not None:
                _rl_url = (page_snapshot.url if page_snapshot and page_snapshot.url else "")
                _rl_hint = self._rl.get_affordances_hint(_rl_url, self.goal)
                if _rl_hint:
                    lines.append("")
                    lines.append(_rl_hint)
        except Exception:
            pass

        return "\n".join(lines)

    def _build_history_text(self) -> str:
        """Build a text summary of all previous steps for the agent_history section."""
        if not self.history:
            return ""

        lines = []

        # Include accumulated scratchpad (persistent context across the run)
        if self.scratchpad:
            lines.append("=== SCRATCHPAD (accumulated findings) ===")
            for note in self.scratchpad[-15:]:  # Last 15 entries
                lines.append(f"  {note}")
            lines.append("=== END SCRATCHPAD ===\n")

        # Include failure tracking context
        if self._consecutive_failures >= 2:
            lines.append(
                f"\n> [!RETRY ALERT] You have FAILED {self._consecutive_failures} times "
                f"on the same goal. You MUST try a DIFFERENT approach now.\n"
                f"> Strategies to try: different coordinates, keyboard shortcut instead of "
                f"click, scroll to reveal element, navigate to a different page, or skip "
                f"this sub-task and move on.\n"
            )

        # Show detailed recent steps + compact summary for older ones
        recent_entries = self.history[-15:]
        total = len(recent_entries)
        for idx, entry in enumerate(recent_entries):
            is_recent = idx >= total - 3   # Last 3 steps always get full detail
            is_failed = any(not r.success for r in entry.action_results)
            ao = entry.agent_output

            if is_recent or is_failed:
                # Full detail for recent/failed steps
                text = entry.to_history_text()
                if ao.memory:
                    text += f"\n  Memory: {ao.memory[:200]}"
                if hasattr(ao, 'confidence') and ao.confidence != "high":
                    text += f"\n  Confidence: {ao.confidence}"
            else:
                # Compact summary for older succeeded steps — save tokens
                results_summary = "; ".join(
                    "✅" if r.success else f"❌ {(r.error or '')[:40]}"
                    for r in entry.action_results
                )
                text = f"Step {entry.step_number + 1}: {ao.next_goal[:80]} [{results_summary}]"
                if ao.memory and len(ao.memory) > 30:
                    text += f" | Memory: {ao.memory[:80]}"

            lines.append(text)

        # --- Stall / Loop Detection (two strategies) ---
        if len(self.history) >= 3:
            last_3 = self.history[-3:]

            # Strategy 1: Exact same actions repeated
            acts_0 = [a.model_dump() for a in last_3[0].agent_output.action]
            acts_1 = [a.model_dump() for a in last_3[1].agent_output.action]
            acts_2 = [a.model_dump() for a in last_3[2].agent_output.action]
            exact_loop = (acts_0 == acts_1 == acts_2)

            # Strategy 2: Same goal repeated (even if coordinates differ slightly)
            goals = [e.agent_output.next_goal.strip().lower() for e in last_3]
            goal_loop = (goals[0] == goals[1] == goals[2])

            # Strategy 3: Similar goals (fuzzy — first 40 chars match)
            goal_prefixes = [g[:40] for g in goals]
            fuzzy_loop = (goal_prefixes[0] == goal_prefixes[1] == goal_prefixes[2])

            if exact_loop or goal_loop or fuzzy_loop:
                loop_type = "exact same actions" if exact_loop else "same goal/intent"
                logger.warning(f"🔄 Loop detected ({loop_type}): Agent stuck for 3 steps.")
                lines.append(
                    "\n> [!CRITICAL WARNING]\n"
                    "> ⚠️ STALL DETECTED ⚠️\n"
                    f"> You have been repeating the same thing ({loop_type}) for the last 3 steps.\n"
                    "> STOP doing what you're doing. You MUST:\n"
                    "> 1. READ the screenshot carefully — what does it actually show right now?\n"
                    "> 2. Check your memory — what questions have you already asked?\n"
                    "> 3. Ask a COMPLETELY DIFFERENT question, or try a different approach entirely.\n"
                    "> 4. If you've completed enough of the task, call `done` with your findings.\n"
                    "> DO NOT repeat the same goal or ask the same question again."
                )

        return "\n".join(lines)

    async def _compress_history(self) -> None:
        """
        Compress old history entries into a heuristic summary to prevent context overflow.

        Keeps the last 10 steps verbatim. Older steps are compressed into:
        - Distinct goals achieved (successes)
        - Distinct failures encountered (for avoiding repetition)
        - URL changes (navigation progress)

        Uses heuristic compression (no LLM call) to avoid nested API latency.
        Called every AUTOBOT_COMPRESS_INTERVAL steps (default 25).
        """
        keep = 10
        old_entries = self.history[:-keep]
        if not old_entries:
            return

        # Extract key signals from old history
        successes: list[str] = []
        failures: list[str] = []
        urls_visited: list[str] = []
        seen_goals: set[str] = set()

        for e in old_entries:
            goal_key = e.agent_output.next_goal.strip().lower()[:60]
            if goal_key in seen_goals:
                continue
            seen_goals.add(goal_key)

            all_success = all(r.success for r in e.action_results)
            any_nav = e.url_before != e.url_after and e.url_after

            if any_nav and e.url_after not in urls_visited:
                urls_visited.append(e.url_after.split("?")[0][:50])

            if all_success:
                successes.append(e.agent_output.next_goal[:60])
            else:
                first_error = next(
                    (r.error for r in e.action_results if r.error), None
                )
                failures.append(f"{e.agent_output.next_goal[:50]}: {(first_error or 'failed')[:40]}")

        # Build compact summary
        parts = []
        first_step = old_entries[0].step_number + 1
        last_step = old_entries[-1].step_number + 1
        parts.append(f"Steps {first_step}–{last_step} summary:")

        if urls_visited:
            parts.append(f"  Navigated to: {', '.join(urls_visited[-5:])}")
        if successes:
            parts.append(f"  Completed: {'; '.join(successes[-8:])}")
        if failures:
            parts.append(f"  Failed (don't repeat): {'; '.join(failures[-5:])}")

        summary = "\n".join(parts)

        # Inject into scratchpad and trim old history
        self.scratchpad.append(f"[HISTORY SUMMARY]\n{summary}")
        self.history = self.history[-keep:]
        logger.info(
            f"📜 History compressed: {len(old_entries)} entries → summary. "
            f"{len(successes)} successes, {len(failures)} failures, {len(urls_visited)} URLs."
        )

    def _summarize_history(self) -> str:
        """Generate a comprehensive summary when the agent hits max steps."""
        if not self.history:
            return "No steps were executed."

        total = len(self.history)
        successes = sum(1 for e in self.history if all(r.success for r in e.action_results))
        failures = total - successes

        # Collect URLs visited (deduplicated)
        urls = []
        seen_urls = set()
        for e in self.history:
            u = (e.url_after or "").split("?")[0]
            if u and u not in seen_urls:
                seen_urls.add(u)
                urls.append(u)

        # Collect substantive scratchpad entries (skip generic filler)
        _skip_prefixes = ("[RECOVERY", "[FORCED", "[RETRY ALERT", "[WATCHDOG")
        key_findings = [
            s for s in self.scratchpad
            if len(s) > 30 and not any(skip in s for skip in _skip_prefixes)
        ][-6:]

        # Last 3 steps for context
        last_steps = [entry.to_history_text() for entry in self.history[-3:]]

        lines = [
            f"Agent completed {total} steps (budget exhausted): "
            f"{successes} succeeded, {failures} failed.",
        ]
        if urls:
            lines.append(f"Pages visited: {', '.join(urls[-5:])}")
        if key_findings:
            lines.append(f"Key findings:\n" + "\n".join(f"  {f}" for f in key_findings))
        lines.append(f"Last 3 steps:\n" + "\n".join(last_steps))

        return "\n".join(lines)

def _infer_page_type_hint(page_snapshot: Any) -> str | None:
    """
    Classify the current page and return a tool recommendation hint.

    Looks at the URL, title, and page text to identify common page types
    (login, form, table/data, search results, code editor, error page)
    and suggests the most effective tools for each.

    Returns a concise one-line hint string, or None if no strong match.
    """
    url = (page_snapshot.url or "").lower()
    title = (page_snapshot.title or "").lower()
    text = (page_snapshot.text or "").lower()[:500]

    # Login / auth page
    if any(w in url + title + text for w in ("login", "sign in", "signin", "log in", "authenticate", "password")):
        return "Page type: LOGIN — Use keyboard.type() for username/password fields. Click 'Sign in' button to submit."

    # Registration / sign-up page
    if any(w in url + title + text for w in ("register", "sign up", "signup", "create account", "join")):
        return "Page type: REGISTRATION — Fill each field with keyboard.type() after clicking it. Use Tab to move between fields."

    # Search results page
    if any(w in url for w in ("search", "results", "q=", "query=")) or "search results" in title:
        return "Page type: SEARCH RESULTS — Scan result titles in the DOM snapshot. Click the most relevant result by index."

    # Data table / listing page
    if any(w in title + text for w in ("table", "spreadsheet", "dataset", "leaderboard", "rankings")):
        return "Page type: DATA TABLE — Use Ctrl+A then Ctrl+C to copy table data, or extract row-by-row using DOM indices."

    # Code editor (VS Code, GitHub, Jupyter, CodePen)
    if any(w in url + title for w in ("github.com", "vscode", "colab", "jupyter", "codepen", "replit", "leetcode")):
        return "Page type: CODE EDITOR — Use keyboard shortcuts (Ctrl+A select all, Ctrl+C copy, Ctrl+V paste). Focus editor area first."

    # Error page
    if any(w in title + text for w in ("404", "not found", "error", "page not found", "403", "forbidden")):
        return "Page type: ERROR PAGE — Navigate away using the `navigate` action. Do not interact with this page."

    # File upload page
    if any(w in title + text for w in ("upload", "drop file", "choose file", "browse file")):
        return "Page type: FILE UPLOAD — Use computer.files to locate the file path, then interact with the file input."

    return None


# Alias for compatibility with the backend (app.py)
AgentRunner = AgentLoop
