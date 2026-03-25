"""
Agent Runner — Manages the lifecycle of an agent task (browser + LLM + loop).

This is the top-level entry point that the dashboard API calls.
It handles:
    1. Launching Chrome via CDP
    2. Setting up the OpenAI client with the user's LLM config
    3. Running the AgentLoop
    4. Logging step-by-step progress
    5. Cleanup on completion or error

Usage:
    runner = AgentRunner.from_env()
    result = await runner.run("search for AI papers on arxiv")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from autobot.agent.loop import AgentLoop
from autobot.agent.mission_agent import MissionAgent
from autobot.agent.planner import ComplexityEstimator
from autobot.browser.launcher import AsyncBrowserLauncher

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Top-level runner that manages browser + LLM + agent loop lifecycle.

    The dashboard API creates one of these per task and calls run().
    """

    def __init__(
        self,
        browser_launcher: AsyncBrowserLauncher | None = None,
        llm_client: Any | None = None,
        model: str = "gpt-4o",
        max_steps: int = 100,
        use_vision: bool = True,
        log_callback: Callable[[str], None] | None = None,
        task_id: str | None = None,
    ):
        self.browser_launcher = browser_launcher or AsyncBrowserLauncher.from_env()
        self.llm_client = llm_client
        self.model = model
        # Fast model for routine steps — cheaper/faster (e.g. gemini-2.0-flash)
        # Set AUTOBOT_FAST_MODEL env var to enable multi-LLM routing.
        # If unset, all steps use the primary model.
        self.fast_model: str | None = os.getenv("AUTOBOT_FAST_MODEL")
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.log = log_callback or (lambda msg: logger.info(msg))
        self.task_id = task_id          # forwarded to AgentLoop → ScreenLock
        self._last_screenshot_path_fallback: str | None = None

        # State tracking for dashboard
        self.status: str = "idle"  # idle | starting | running | done | failed
        self.current_step: int = 0
        self.current_goal: str = ""
        self.result: str = ""
        self._agent_loop: AgentLoop | None = None
        self._mission_agent: MissionAgent | None = None
        self._max_steps_override: int | None = None

    @classmethod
    def from_env(
        cls,
        log_callback: Callable[[str], None] | None = None,
        task_id: str | None = None,
    ) -> "AgentRunner":
        """Create runner from environment variables."""
        llm_client = _create_llm_client()
        model = os.getenv("AUTOBOT_LLM_MODEL", "gpt-4o")

        return cls(
            llm_client=llm_client,
            model=model,
            log_callback=log_callback,
            task_id=task_id,
        )

    async def run(self, goal: str, max_steps: int | None = None) -> str:
        """
        Run a task end-to-end: launch browser → run agent loop → return result.

        Args:
            goal: Natural language task description.
            max_steps: Override max steps (default: self.max_steps).

        Returns:
            Result text from the agent.
        """
        self.status = "starting"
        self.current_goal = goal
        self.current_step = 0
        self._max_steps_override = max_steps or self.max_steps
        steps = self._max_steps_override

        self.log(f"🤖 Starting task: {goal}")

        try:
            # 1. Create LLM client and run pre-flight check
            if self.llm_client is None:
                self.llm_client = _create_llm_client()
                if self.llm_client is None:
                    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "openrouter")
                    key_hint = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
                    raise RuntimeError(
                        f"No LLM API key found. Set {key_hint} in your .env file. "
                        f"Current provider: {provider}"
                    )

            # 2. Estimate task complexity — sets step budget and stop condition
            self.log("🧠 Estimating task complexity...")
            estimator = ComplexityEstimator(llm_client=self.llm_client, model=self.model)
            estimate = await estimator.estimate(goal)
            # Allow explicit max_steps override from caller
            if max_steps is not None:
                from autobot.agent.stop_condition import after_steps
                estimate.stop_condition = after_steps(max_steps)
                estimate.step_budget = max_steps
            self._max_steps_override = estimate.step_budget
            steps = estimate.step_budget
            self.log(f"📋 Mode: {estimate.mode} | Budget: {'∞' if not steps else steps} steps | {estimate.stop_condition.description}")
            self.log(f"   Reasoning: {estimate.reasoning}")

            self.log(f"🔍 Pre-flight: testing {self.model} ...")
            ok = await self._preflight_check()
            if not ok:
                self.log(f"⚠️  Pre-flight failed — model '{self.model}' may be unavailable or rate-limited. "
                         f"The agent will still attempt to run with fallback strategies.")

            # 3. Launch browser
            self.log("Initializing Human Mode (Vision-Only)...")
            page = await self.browser_launcher.start()
            self.log("✅ Human Mode active. Operating in your real Chrome profile.")

            # 4. Create run directory for checkpoints and artifacts
            runs_dir = Path("runs")
            runs_dir.mkdir(exist_ok=True)
            run_dir = runs_dir / f"plan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "screenshots").mkdir(exist_ok=True)
            (run_dir / "about.txt").write_text(
                f"Goal: {goal}\nModel: {self.model}\nMode: {estimate.mode}\n"
                f"Budget: {steps or 'perpetual'} steps\nStop: {estimate.stop_condition.description}\n"
            )

            # 5. Create and run agent loop with dynamic stop condition
            self.status = "running"
            self.current_step = 1
            self._agent_loop = AgentLoop(
                page=page,
                llm_client=self.llm_client,
                goal=goal,
                model=self.model,
                fast_model=self.fast_model,
                max_steps=steps,
                use_vision=self.use_vision,
                stop_condition=estimate.stop_condition,
                task_id=self.task_id,
            )
            self._agent_loop._run_dir = run_dir  # Enable checkpointing

            # Hook into the agent loop to track progress
            original_execute_step = self._agent_loop._execute_step

            async def _tracked_execute_step() -> str | None:
                self.current_step = self._agent_loop.step_number + 1
                self.log(f"📍 Step {self.current_step}/{steps}")
                
                # Double check LLM client hasn't vanished
                if getattr(self._agent_loop, "llm_client", None) is None:
                    raise RuntimeError("Agent loop lost LLM client connection unexpectedly.")
                    
                result = await original_execute_step()

                # Log the agent's thinking and actions in detail
                if self._agent_loop.history:
                    last = self._agent_loop.history[-1]
                    ao = last.agent_output
                    confidence = getattr(ao, 'confidence', 'high')
                    conf_icon = "🟢" if confidence == "high" else "🟡" if confidence == "medium" else "🔴"
                    self.log(f"  🧠 Thinking: {ao.thinking[:120]}...")
                    self.log(f"  💭 Goal: {ao.next_goal}")
                    self.log(f"  {conf_icon} Confidence: {confidence}")
                    if ao.memory:
                        self.log(f"  📝 Memory: {ao.memory[:100]}")
                    # Show retry tracking
                    if self._agent_loop._consecutive_failures >= 2:
                        self.log(f"  🔄 Retry alert: {self._agent_loop._consecutive_failures} consecutive failures")
                    for action, ar in zip(ao.action, last.action_results):
                        icon = "✅" if ar.success else "❌"
                        detail = ar.action_name
                        if action.computer_call:
                            detail = action.computer_call.call
                        elif action.navigate:
                            detail = f"navigate → {action.navigate.url[:60]}"
                        elif action.wait:
                            detail = f"wait({action.wait.seconds}s)"
                        elif action.done:
                            detail = f"done(success={action.done.success})"
                        self.log(f"  {icon} {detail}")
                        if ar.error:
                            self.log(f"     ⚠️ {ar.error[:100]}")

                return result

            self._agent_loop._execute_step = _tracked_execute_step

            loop_result = await self._agent_loop.run()
            
            # 4. Evaluate with Judge Agent
            self.log("⚖️ Judge Agent evaluating outcome...")
            from autobot.agent.judge import JudgeAgent
            judge = JudgeAgent(llm_client=self.llm_client, model=self.model)
            
            history_summary = self._agent_loop._build_history_text()
            judge_output = await judge.evaluate(
                goal=goal,
                result_text=loop_result,
                history_summary=history_summary
            )
            
            if judge_output.success:
                self.log(f"🏆 Judge confirmed success: {judge_output.reasoning}")
                self.status = "done"
                self.result = f"{loop_result}\n\n[Judge Verification: SUCCESS] {judge_output.reasoning}"
            else:
                self.log(f"📛 Judge reported failure: {judge_output.reasoning}")
                self.status = "failed"
                self.result = f"{loop_result}\n\n[Judge Verification: FAILED] {judge_output.reasoning}"

            self.log(f"🏁 Task finished. Result: {self.result[:200]}")
            return self.result

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.status = "failed"
            self.result = f"Error: {e}\n\nTraceback:\n{tb}"
            self.log(f"❌ Task failed: {e}\n{tb}")
            raise

        finally:
            # Disconnect Playwright (Chrome stays running)
            try:
                await self.browser_launcher.stop()
            except Exception:
                pass

    async def _preflight_check(self) -> bool:
        """
        Send a minimal test message to verify the LLM is reachable and responding.
        Returns True if the model replies, False on any error.
        """
        try:
            resp = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                max_tokens=16,   # Match real call cap; a failed pre-flight here means real calls will also fail
                temperature=0,
            )
            if resp.choices and resp.choices[0].message.content:
                self.log(f"✅ LLM pre-flight OK ({self.model})")
                return True
            self.log(f"⚠️  LLM pre-flight: empty response from {self.model}")
            return False
        except Exception as e:
            self.log(f"⚠️  LLM pre-flight FAILED: {e}")
            return False

    async def run_mission(self, goal: str) -> str:
        """
        Run a complex multi-objective mission.

        Uses MissionAgent to break the goal into objectives and execute each one.
        Best for tasks like: Kaggle competitions, research workflows, multi-app coding.
        """
        self.status = "starting"
        self.current_goal = goal
        self.current_step = 0

        self.log(f"🚀 Starting Mission: {goal}")

        try:
            if self.llm_client is None:
                self.llm_client = _create_llm_client()
                if self.llm_client is None:
                    raise RuntimeError("No LLM API key found.")

            # Launch browser
            self.log("Initializing Human Mode (Vision-Only)...")
            page = await self.browser_launcher.start()
            self.log("✅ Human Mode active.")

            # Create run directory
            runs_dir = Path("runs")
            runs_dir.mkdir(exist_ok=True)
            run_dir = runs_dir / f"mission_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "about.txt").write_text(f"Mission: {goal}\nModel: {self.model}\n")

            self.status = "running"
            self._mission_agent = MissionAgent(
                page=page,
                llm_client=self.llm_client,
                mission_goal=goal,
                model=self.model,
                log_callback=self.log,
            )
            self._mission_agent._run_dir = run_dir

            result = await self._mission_agent.run()

            # Track the current agent loop for screenshots
            if self._mission_agent.current_agent_loop:
                self._agent_loop = self._mission_agent.current_agent_loop

            self.status = "done"
            self.result = result
            self.log(f"🏁 Mission finished: {result[:200]}")
            return result

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.status = "failed"
            self.result = f"Error: {e}\n\n{tb}"
            self.log(f"❌ Mission failed: {e}")
            raise
        finally:
            try:
                await self.browser_launcher.stop()
            except Exception:
                pass

    def pause(self) -> None:
        """Pause the running task after the current step completes."""
        self.status = "paused"
        if self._agent_loop:
            self._agent_loop.pause()
        if hasattr(self, '_mission_agent') and self._mission_agent and self._mission_agent.current_agent_loop:
            self._mission_agent.current_agent_loop.pause()
        self.log("⏸ Task paused")

    def resume(self) -> None:
        """Resume a paused task."""
        self.status = "running"
        if self._agent_loop:
            self._agent_loop.resume()
        if hasattr(self, '_mission_agent') and self._mission_agent and self._mission_agent.current_agent_loop:
            self._mission_agent.current_agent_loop.resume()
        self.log("▶ Task resumed")

    def cancel(self) -> None:
        """Cancel the running task immediately.

        Calls AgentLoop.cancel() which both sets the cancellation flag AND
        cancels any in-flight LLM asyncio Task, stopping execution within
        milliseconds rather than waiting for the next loop iteration.
        """
        self.status = "cancelled"
        if self._agent_loop:
            self._agent_loop.cancel()
        # Also cancel mission agent's current loop
        if hasattr(self, '_mission_agent') and self._mission_agent and self._mission_agent.current_agent_loop:
            self._mission_agent.current_agent_loop.cancel()
        self.log("⚠️ Task cancelled")

    @property
    def last_screenshot_path(self) -> str | None:
        """Get the last screenshot path from the current agent loop."""
        # Check mission agent's current loop first
        if hasattr(self, '_mission_agent') and self._mission_agent and self._mission_agent.current_agent_loop:
            return self._mission_agent.current_agent_loop.last_screenshot_path
        if self._agent_loop:
            return self._agent_loop.last_screenshot_path
        return getattr(self, "_last_screenshot_path_fallback", None)

    @last_screenshot_path.setter
    def last_screenshot_path(self, value: str | None) -> None:
        self._last_screenshot_path_fallback = value

    def get_status(self) -> dict[str, Any]:
        """Get current runner status for the dashboard API."""
        # Determine the active agent loop (could be from mission agent)
        active_loop = self._agent_loop
        if hasattr(self, '_mission_agent') and self._mission_agent and self._mission_agent.current_agent_loop:
            active_loop = self._mission_agent.current_agent_loop

        # Pull live state from active loop
        loop_status = active_loop.get_status() if active_loop else {}

        # Inline screenshot as base64 for extension popup / status polling
        screenshot_b64: str | None = None
        ss_path = self.last_screenshot_path
        if ss_path:
            from pathlib import Path as _Path
            p = _Path(ss_path)
            if p.exists():
                import base64 as _b64
                screenshot_b64 = _b64.b64encode(p.read_bytes()).decode()

        status = {
            "status": self.status,
            "goal": self.current_goal,
            "current_step": self.current_step,
            "max_steps": loop_status.get("max_steps") or self._max_steps_override or self.max_steps,
            "result": self.result[:500] if self.result else "",
            # Screenshot (base64 JPEG) for live dashboard / extension
            "screenshot_b64": screenshot_b64,
            # Plain-English narrative from agent
            "narrative": loop_status.get("narrative", ""),
            # LLM info
            "llm_provider": os.getenv("AUTOBOT_LLM_PROVIDER", ""),
            "llm_model": self.model,
            # Current browser URL (best-effort — page may be None if not yet launched)
            "browser_url": self._get_current_url(),
            # Evaluation + stop condition progress
            "paused": loop_status.get("paused", False),
            "eval_signal": loop_status.get("eval_signal", "continue"),
            "stop_condition": loop_status.get("stop_condition"),
            "stop_progress": loop_status.get("stop_progress", ""),
            "metrics": loop_status.get("metrics", {}),
            "elapsed_seconds": loop_status.get("elapsed_seconds", 0),
            "history": [
                entry.to_history_text()
                for entry in (active_loop.history if active_loop else [])
            ][-10:],
        }
        # Auth / escalation notification
        if active_loop and active_loop.pending_auth:
            status["auth_notification"] = active_loop.pending_auth
        # Mission status if running a mission
        if hasattr(self, '_mission_agent') and self._mission_agent:
            status["mission"] = self._mission_agent.get_status()
        return status

    def _get_current_url(self) -> str:
        """Best-effort: return the current page URL from the active agent loop."""
        try:
            loop = self._agent_loop
            if loop and loop.page:
                return loop.page.url or ""
        except Exception:
            pass
        return ""


def _create_llm_client() -> Any | None:
    """
    Create an async OpenAI-compatible client from environment variables.

    Supports:
    - google       → Google Gemini via OpenAI-compatible endpoint (GOOGLE_API_KEY)
    - openrouter   → OpenRouter (OPENROUTER_API_KEY)
    - openai       → OpenAI directly (OPENAI_API_KEY)
    - xai          → Grok via x.ai (XAI_API_KEY)
    - auto         → tries google → openrouter → openai in order

    Default key: if no user key is set, falls back to AUTOBOT_DEFAULT_API_KEY
    (bundled in the distributed binary for zero-config first-run experience).
    """
    from openai import AsyncOpenAI

    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "auto").lower()

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not set. Add it to your .env file.")
            return None
        # Google exposes an OpenAI-compatible endpoint for Gemini models
        return AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return AsyncOpenAI(api_key=api_key)

    elif provider == "xai":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            return None
        return AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key,
        )

    else:
        # Auto-detect: try in order google → openrouter → openai
        for env_key, base_url in [
            ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ]:
            api_key = os.getenv(env_key)
            if api_key:
                return AsyncOpenAI(base_url=base_url, api_key=api_key)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return AsyncOpenAI(api_key=api_key)

        # ── Default key fallback (zero-config / bundled binary) ──────────────
        # AUTOBOT_DEFAULT_API_KEY and AUTOBOT_DEFAULT_PROVIDER are set when
        # building the distributed binary so users can start immediately without
        # configuring their own key. Users override by adding their own key to
        # .env or via the Settings page.
        default_key = os.getenv("AUTOBOT_DEFAULT_API_KEY")
        default_provider = os.getenv("AUTOBOT_DEFAULT_PROVIDER", "google")
        default_model = os.getenv("AUTOBOT_DEFAULT_MODEL", "gemini-2.0-flash")
        if default_key:
            logger.info(
                f"Using built-in shared API key (provider={default_provider}, model={default_model}). "
                "Add your own key in Settings to avoid rate limits."
            )
            # Set env vars so the rest of the system picks up the right model
            if not os.getenv("AUTOBOT_LLM_MODEL"):
                os.environ["AUTOBOT_LLM_MODEL"] = default_model
            if default_provider == "google":
                return AsyncOpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=default_key,
                )
            elif default_provider == "openrouter":
                return AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=default_key,
                )
            return AsyncOpenAI(api_key=default_key)

        return None
