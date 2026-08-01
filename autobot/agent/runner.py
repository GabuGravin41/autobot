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
        task_id: str | None = None,
    ):
        self.browser_launcher = browser_launcher or AsyncBrowserLauncher.from_env()
        self.llm_client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.log = log_callback or (lambda msg: logger.info(msg))
        self.task_id = task_id

        # State tracking for dashboard
        self.status: str = "idle"  # idle | starting | running | done | failed
        self.current_step: int = 0
        self.current_goal: str = ""
        self.result: str = ""
        self._agent_loop: AgentLoop | None = None
        self._mission_agent: Any | None = None

    @classmethod
    def from_env(
        cls,
        log_callback: Callable[[str], None] | None = None,
        task_id: str | None = None,
    ) -> "AgentRunner":
        """Create runner from environment variables."""
        llm_client = _create_llm_client()
        provider = os.getenv("AUTOBOT_LLM_PROVIDER", "").lower()
        has_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if provider == "gemini" or (has_gemini and not os.getenv("OPENROUTER_API_KEY")):
            default_model = "gemini-1.5-flash"
        else:
            default_model = "gpt-4o"

        model = os.getenv("AUTOBOT_LLM_MODEL") or default_model

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
        steps = max_steps or self.max_steps

        self.log(f"🤖 Starting task: {goal}")
        self.log(f"📋 Max steps: {steps} | Model: {self.model}")

        try:
            # 1. Attach a browser IF WE CAN — but never let its absence end the
            # run. Plenty of goals ("open Notepad and type hello", "run this
            # script", "focus Artemis and read the annotation") involve no
            # browser at all, and this used to raise and kill the task before
            # a single agent step executed. Autobot is a computer-use agent,
            # not a browser-only agent: no Chrome means reduced capability,
            # not failure.
            self.log("🌐 Attaching to Chrome via CDP...")
            page = None
            try:
                page = await self.browser_launcher.start()
                self.log(f"✅ Browser connected. Current page: {page.url}")
            except Exception as browser_error:
                self.log(
                    "⚠️  No browser attached - continuing in OS-only mode "
                    "(native apps, mouse/keyboard, terminal, files still work).\n"
                    f"    Reason: {str(browser_error).splitlines()[0]}"
                )
                logger.warning(f"Browser attach failed, degrading to OS-only: {browser_error}")

            # 2. Create LLM client if not provided
            if self.llm_client is None:
                self.llm_client = _create_llm_client()
                if self.llm_client is None:
                    raise RuntimeError(
                        "No LLM API key configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"
                    )

            # 3. Route by complexity. A goal like "open Chrome, hold a 6-turn
            # conversation on Grok, switch tabs, create+name an Overleaf
            # project, paste LaTeX, recompile" has multiple independent
            # phases — previously this ALWAYS went through one flat AgentLoop
            # sharing a single step budget across the entire goal, with no
            # phase boundaries. Multi-phase goals now get MissionAgent's
            # objective decomposition (its own step budget per phase);
            # single-phase goals keep the original flat-loop path unchanged.
            self.status = "running"
            from autobot.agent.orchestrator import TaskClassifier

            if TaskClassifier.is_complex(goal):
                self.log("🔀 Multi-phase goal detected — using MissionAgent objective decomposition")
                loop_result, history_summary = await self._run_mission(page, goal, steps)
            else:
                loop_result, history_summary = await self._run_single_loop(page, goal, steps)

            # 4. Evaluate with Judge Agent
            self.log("⚖️ Judge Agent evaluating outcome...")
            from autobot.agent.judge import JudgeAgent
            judge = JudgeAgent(llm_client=self.llm_client, model=self.model)

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

    async def _run_single_loop(self, page: Any, goal: str, steps: int) -> tuple[str, str]:
        """Run a single-phase goal through one flat AgentLoop (original behavior)."""
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
        history_summary = self._agent_loop._build_history_text()
        return loop_result, history_summary

    async def _run_mission(self, page: Any, goal: str, steps: int) -> tuple[str, str]:
        """Run a multi-phase goal through MissionAgent's objective decomposition.

        Each objective gets its own AgentLoop and step budget (default `steps`,
        or the planner's own per-objective estimate). MissionAgent tracks the
        currently-running AgentLoop on `current_agent_loop` — we mirror that
        onto self._agent_loop after each objective so get_status() and
        cancel() keep working exactly as they do for the single-loop path.
        """
        from autobot.agent.mission_agent import MissionAgent

        mission = MissionAgent(
            page=page,
            llm_client=self.llm_client,
            mission_goal=goal,
            model=self.model,
            max_steps_per_objective=steps,
            log_callback=self.log,
        )
        self._mission_agent = mission

        loop_result = await mission.run()
        # Mirror MissionAgent's active AgentLoop so dashboard status/cancel
        # keep reflecting the last objective that ran.
        self._agent_loop = mission.current_agent_loop
        if self._agent_loop is not None:
            self.current_step = self._agent_loop.step_number + 1

        history_summary = mission._get_mission_summary()
        return loop_result, history_summary

    def cancel(self) -> None:
        """Cancel the running task."""
        self.status = "cancelled"
        if self._agent_loop:
            # Set max_steps to 0 to stop the current AgentLoop
            self._agent_loop.max_steps = 0
        if self._mission_agent:
            # Also stop MissionAgent from advancing to the next objective
            from autobot.agent.mission import MissionStatus
            self._mission_agent.mission.status = MissionStatus.FAILED
        self.log("⚠️ Task cancelled")

    def push_override(self, new_instruction: str) -> None:
        """Push mid-flight intervention to active agent loop."""
        if self._agent_loop:
            self._agent_loop.push_override(new_instruction)
            self.log(f"⚡ Mid-flight pivot pushed: {new_instruction}")

    def get_status(self) -> dict[str, Any]:
        """Get current runner status for the dashboard API."""
        return {
            "status": self.status,
            "goal": self.current_goal,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
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
    - Anthropic Claude (ANTHROPIC_API_KEY)
    - OpenRouter (OPENROUTER_API_KEY) — default
    - OpenAI (OPENAI_API_KEY)
    - Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
    """
    from openai import OpenAI
    from autobot.agent.anthropic_adapter import get_anthropic_llm_client

    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "auto").lower()

    if provider == "anthropic":
        return get_anthropic_llm_client()

    elif provider == "openrouter":
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

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )

    else:
        # Fallback priority: Anthropic -> OpenRouter -> Gemini (Google AI Studio) -> OpenAI
        anth_client = get_anthropic_llm_client()
        if anth_client:
            return anth_client

        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            return OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key,
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None
