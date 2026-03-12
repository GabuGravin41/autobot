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
from typing import Any, Callable

from autobot.agent.loop import AgentLoop
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
        max_steps: int = 25,
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
            # 1. Launch browser
            self.log("🌐 Launching Chrome with CDP...")
            page = await self.browser_launcher.start()
            self.log(f"✅ Browser connected. Current page: {page.url}")

            # 2. Create LLM client if not provided
            if self.llm_client is None:
                self.llm_client = _create_llm_client()
                if self.llm_client is None:
                    raise RuntimeError(
                        "No LLM API key configured. Please create a '.env' file in the autobot root directory "
                        "and set either OPENROUTER_API_KEY=... or OPENAI_API_KEY=..."
                    )

            # 3. Create and run agent loop
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

            # Hook into the agent loop to track progress
            original_execute_step = self._agent_loop._execute_step

            async def _tracked_execute_step() -> str | None:
                self.current_step = self._agent_loop.step_number + 1
                self.log(f"📍 Step {self.current_step}/{steps}")
                
                # Double check LLM client hasn't vanished
                if getattr(self._agent_loop, "llm_client", None) is None:
                    raise RuntimeError("Agent loop lost LLM client connection unexpectedly.")
                    
                result = await original_execute_step()

                # Log the agent's thinking
                if self._agent_loop.history:
                    last = self._agent_loop.history[-1]
                    self.log(f"  💭 {last.agent_output.next_goal}")
                    for ar in last.action_results:
                        icon = "✅" if ar.success else "❌"
                        self.log(f"  {icon} {ar.action_name}")
                        if ar.error:
                            self.log(f"     Error: {ar.error}")

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

    def cancel(self) -> None:
        """Cancel the running task."""
        self.status = "cancelled"
        if self._agent_loop:
            # Set max_steps to 0 to stop the loop
            self._agent_loop.max_steps = 0
        self.log("⚠️ Task cancelled")

    @property
    def last_screenshot_path(self) -> str | None:
        """Get the last screenshot path from the current agent loop."""
        if self._agent_loop:
            return self._agent_loop.last_screenshot_path
        return getattr(self, "_last_screenshot_path_fallback", None)

    @last_screenshot_path.setter
    def last_screenshot_path(self, value: str | None) -> None:
        self._last_screenshot_path_fallback = value

    def get_status(self) -> dict[str, Any]:
        """Get current runner status for the dashboard API."""
        return {
            "status": self.status,
            "goal": self.current_goal,
            "current_step": self.current_step,
            "max_steps": self._max_steps_override or self.max_steps,
            "result": self.result[:500] if self.result else "",
            "history": [
                entry.to_history_text()
                for entry in (self._agent_loop.history if self._agent_loop else [])
            ][-5:],  # Last 5 steps
        }


def _create_llm_client() -> Any | None:
    """
    Create an OpenAI-compatible client from environment variables.

    Supports:
    - OpenRouter (OPENROUTER_API_KEY) — default
    - OpenAI (OPENAI_API_KEY)
    - Any OpenAI-compatible API
    """
    from openai import OpenAI

    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "openrouter").lower()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)

    else:
        # Try OpenRouter first, then OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None
