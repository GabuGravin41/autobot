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
    ):
        self.browser_launcher = browser_launcher or AsyncBrowserLauncher.from_env()
        self.llm_client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.log = log_callback or (lambda msg: logger.info(msg))
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
    def from_env(cls, log_callback: Callable[[str], None] | None = None) -> "AgentRunner":
        """Create runner from environment variables."""
        llm_client = _create_llm_client()
        model = os.getenv("AUTOBOT_LLM_MODEL", "gpt-4o")

        return cls(
            llm_client=llm_client,
            model=model,
            log_callback=log_callback,
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
        self.log(f"📋 Max steps: {steps} | Model: {self.model}")

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

            self.log(f"🔍 Pre-flight: testing {self.model} ...")
            ok = await self._preflight_check()
            if not ok:
                self.log(f"⚠️  Pre-flight failed — model '{self.model}' may be unavailable or rate-limited. "
                         f"The agent will still attempt to run with fallback strategies.")

            # 2. Launch browser
            self.log("Initializing Human Mode (Vision-Only)...")
            page = await self.browser_launcher.start()
            self.log("✅ Human Mode active. Operating in your real Chrome profile.")

            # 3. Create run directory for checkpoints and artifacts
            runs_dir = Path("runs")
            runs_dir.mkdir(exist_ok=True)
            run_dir = runs_dir / f"plan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "screenshots").mkdir(exist_ok=True)
            # Save task description
            (run_dir / "about.txt").write_text(f"Goal: {goal}\nModel: {self.model}\nMax Steps: {steps}\n")

            # 4. Create and run agent loop
            self.status = "running"
            self.current_step = 1
            self._agent_loop = AgentLoop(
                page=page,
                llm_client=self.llm_client,
                goal=goal,
                model=self.model,
                max_steps=steps,
                use_vision=self.use_vision,
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

    def cancel(self) -> None:
        """Cancel the running task."""
        self.status = "cancelled"
        if self._agent_loop:
            self._agent_loop.max_steps = 0
        # Also cancel mission agent's current loop
        if hasattr(self, '_mission_agent') and self._mission_agent and self._mission_agent.current_agent_loop:
            self._mission_agent.current_agent_loop.max_steps = 0
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

        status = {
            "status": self.status,
            "goal": self.current_goal,
            "current_step": self.current_step,
            "max_steps": self._max_steps_override or self.max_steps,
            "result": self.result[:500] if self.result else "",
            "history": [
                entry.to_history_text()
                for entry in (active_loop.history if active_loop else [])
            ][-10:],  # Last 10 steps
        }
        # Include auth notification if agent detected a login page
        if active_loop and active_loop.pending_auth:
            status["auth_notification"] = active_loop.pending_auth
        # Include mission status if running a mission
        if hasattr(self, '_mission_agent') and self._mission_agent:
            status["mission"] = self._mission_agent.get_status()
        return status


def _create_llm_client() -> Any | None:
    """
    Create an async OpenAI-compatible client from environment variables.

    Supports:
    - google       → Google Gemini via OpenAI-compatible endpoint (GOOGLE_API_KEY)
    - openrouter   → OpenRouter (OPENROUTER_API_KEY)
    - openai       → OpenAI directly (OPENAI_API_KEY)
    - xai          → Grok via x.ai (XAI_API_KEY)
    - auto         → tries google → openrouter → openai in order
    """
    from openai import AsyncOpenAI

    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "openrouter").lower()

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
        return None
