"""
Agent Runner — Manages the lifecycle of one agent task.

Simplified from the old runner.py. No browser launcher required upfront,
no MissionAgent routing, no Judge call. Just:

  1. Build a Computer (UIAutomation, keyboard, mouse, etc.)
  2. Create a CoreLoop
  3. Run it
  4. Return the result

The web API (web/app.py) and CLI (cli.py) both call AgentRunner.from_env().
The interface is preserved — from_env(), run(), cancel(), push_override(),
get_status(), current_step, max_steps — so neither caller needs to change.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Top-level runner: Computer + LLM + CoreLoop.

    The dashboard API creates one of these per task and calls run().
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str = "gpt-4o",
        max_steps: int = 25,
        log_callback: Callable[[str], None] | None = None,
        task_id: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.log = log_callback or (lambda msg: logger.info(msg))
        self.task_id = task_id

        # Status tracking for dashboard
        self.status: str = "idle"   # idle | starting | running | done | failed | cancelled
        self.current_step: int = 0
        self.current_goal: str = ""
        self.result: str = ""
        self._loop: Any | None = None   # CoreLoop, set during run()

    @classmethod
    def from_env(
        cls,
        log_callback: Callable[[str], None] | None = None,
        task_id: str | None = None,
    ) -> "AgentRunner":
        """Create a runner from environment variables (.env / os.environ)."""
        llm_client = _create_llm_client()
        model = _default_model()
        return cls(
            llm_client=llm_client,
            model=model,
            log_callback=log_callback,
            task_id=task_id,
        )

    async def run(self, goal: str, max_steps: int | None = None) -> str:
        """
        Run a task end-to-end: UIAutomation → CoreLoop → result.

        Args:
            goal:      Natural language task description.
            max_steps: Override max steps (default: self.max_steps).

        Returns:
            Result text from the agent.
        """
        self.status = "starting"
        self.current_goal = goal
        steps = max_steps or self.max_steps

        # No "Starting.../Max steps..." log line here on purpose — CoreLoop.run()
        # logs the same thing itself (via this same self.log callback) once it's
        # constructed below, and having both log it produced duplicate lines in
        # the run log.

        # Ensure we have an LLM client
        if self.llm_client is None:
            self.llm_client = _create_llm_client()
        if self.llm_client is None:
            msg = (
                "No LLM client available. Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, "
                "or OPENAI_API_KEY in .env. Run 'autobot --doctor' to check your setup."
            )
            self.log(f"❌ {msg}")
            self.status = "failed"
            self.result = msg
            return msg

        try:
            from autobot.computer.computer import Computer
            from autobot.agent.core_loop import CoreLoop

            computer = Computer()

            def _step_log(msg: str) -> None:
                self.log(msg)
                # Mirror the loop's step count
                if self._loop is not None:
                    self.current_step = self._loop.step_number

            self._loop = CoreLoop(
                computer=computer,
                llm_client=self.llm_client,
                goal=goal,
                model=self.model,
                max_steps=steps,
                log=_step_log,
            )
            self.status = "running"
            self.result = await self._loop.run()
            self.status = "done"
            self.log(f"🏁 Finished: {self.result[:200]}")
            return self.result

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.status = "failed"
            self.result = f"Error: {e}"
            self.log(f"❌ Task failed: {e}\n{tb}")
            raise

    def cancel(self) -> None:
        """Cancel the running task."""
        self.status = "cancelled"
        if self._loop is not None:
            self._loop.is_cancelled = True
        self.log("⚠️ Task cancelled")

    def pause(self) -> None:
        """Pause the running task."""
        if self._loop is not None:
            self._loop.pause()
        self.status = "paused"

    def resume(self) -> None:
        """Resume the paused task."""
        if self._loop is not None:
            self._loop.resume()
        self.status = "running"

    def push_override(self, new_instruction: str) -> None:
        """Push a mid-flight goal override to the active loop."""
        if self._loop is not None:
            self._loop.push_override(new_instruction)
            self.log(f"⚡ Override pushed: {new_instruction}")

    def get_status(self) -> dict[str, Any]:
        """Status dict for the dashboard API and task scheduler."""
        loop = self._loop
        return {
            "status": self.status,
            "paused": getattr(loop, "is_paused", False) if loop else False,
            "goal": self.current_goal,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "result": self.result[:500] if self.result else "",
            "history": [
                entry.to_history_text()
                for entry in (loop.history if loop else [])
            ][-5:],
            "eval_signal": (
                "success" if self.status == "done" and (loop._last_done_success if loop else False)
                else "failure" if self.status in ("failed", "cancelled")
                else "continue"
            ),
        }


# ── LLM client factory (unchanged from old runner.py) ────────────────────────

def _default_model() -> str:
    """Infer a sensible default model from env vars."""
    env_model = os.getenv("AUTOBOT_LLM_MODEL", "").strip()
    if env_model:
        return env_model

    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "").lower()
    if provider == "gemini" or (
        (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        and not os.getenv("OPENROUTER_API_KEY")
    ):
        return "gemini-1.5-flash"
    return "openai/gpt-4o-mini"


def _create_llm_client() -> Any | None:
    """
    Create an OpenAI-compatible LLM client from environment variables.

    Supports: Anthropic (ANTHROPIC_API_KEY), OpenRouter (OPENROUTER_API_KEY),
              OpenAI (OPENAI_API_KEY), Gemini (GEMINI_API_KEY / GOOGLE_API_KEY).
    Priority: OpenRouter → Anthropic → Gemini → OpenAI.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(encoding="utf-8-sig")
    except ImportError:
        pass

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None

    def _key(name: str) -> str | None:
        val = os.getenv(name, "").strip()
        return val if val and val.lower() not in ("none", "null", "undefined", "") else None

    provider = os.getenv("AUTOBOT_LLM_PROVIDER", "auto").lower()

    anth_key  = _key("ANTHROPIC_API_KEY")
    or_key    = _key("OPENROUTER_API_KEY")
    oa_key    = _key("OPENAI_API_KEY")
    gem_key   = _key("GEMINI_API_KEY") or _key("GOOGLE_API_KEY")

    if provider == "anthropic" and anth_key:
        from autobot.agent.anthropic_adapter import get_anthropic_llm_client
        return get_anthropic_llm_client(api_key=anth_key)

    if provider == "openrouter" and or_key:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key)

    if provider == "openai" and oa_key:
        return OpenAI(api_key=oa_key)

    if provider == "gemini" and gem_key:
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gem_key,
        )

    # Auto-detect by key priority
    if or_key:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key)
    if anth_key:
        from autobot.agent.anthropic_adapter import get_anthropic_llm_client
        return get_anthropic_llm_client(api_key=anth_key)
    if gem_key:
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gem_key,
        )
    if oa_key:
        return OpenAI(api_key=oa_key)

    # No API key at all: fall back to any OpenAI-compatible endpoint or free
    # tier configured for the butler's manager, and finally to the Claude /
    # Antigravity subscriptions via their CLIs (see autobot/llm/). Slower per
    # step than a direct API, but it means the desktop agent works without a
    # paid key.
    base_url = _key("AUTOBOT_LLM_BASE_URL")
    if base_url:
        return OpenAI(base_url=base_url, api_key=_key("AUTOBOT_LLM_API_KEY") or "not-needed")
    try:
        from autobot.llm import ChatCompletionsShim, get_manager_llm
        chain = get_manager_llm()
        if chain is not None:
            logger.info(f"No API key found; CoreLoop will use {chain.describe()}")
            return ChatCompletionsShim(chain)
    except Exception as e:
        logger.warning(f"LLM fallback unavailable: {e}")
    return None
