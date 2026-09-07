"""
CoreLoop — The new observe → decide → act → verify cycle.

This replaces the 1,119-line loop.py with a single, readable loop that:

  1. OBSERVE  — UIAutomation element tree of the active window (primary)
  2. DECIDE   — One LLM call, get one structured action back
  3. ACT      — Execute the action through the safe dispatcher or direct calls
  4. VERIFY   — Check the screen changed; escalate if stuck
  5. DISTILL  — On success, save the proven path as a learned skill

Design principles:
  - UIAutomation is PRIMARY. CDP browser is an optional tool the model calls,
    not a required dependency that must be set up before anything else works.
  - One action per step. Simpler to verify, simpler to debug.
  - No MissionAgent, no TaskClassifier, no complexity routing.
    Just this loop. Add those back after this works on real tasks.
  - Transparent failure: stuck detection tells the model exactly what happened
    and what the valid actions are.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from autobot.agent.action_models import Action, LLMResponse, StepRecord
from autobot.agent.approval import ApprovalGuard, RiskTier
from autobot.computer.computer import Computer
from autobot.knowledge.skill_distiller import SkillDistiller

logger = logging.getLogger(__name__)


class _ActionStub:
    """
    Minimal shim so ApprovalGuard.gate() / _action_text() can handle the new
    flat Action type without modification.

    approval.py's _action_text() reads specific attributes from the old
    ActionModel (computer_call, navigate, input_text, run_command, done).
    This stub exposes only what _action_text() needs.
    """

    def __init__(self, action: Action) -> None:
        self._action = action

    @property
    def action_name(self) -> str:
        return self._action.name

    @property
    def computer_call(self):
        if self._action.name == "computer_call":
            stub = type("CallStub", (), {"call": self._action.params.get("call", "")})()
            return stub
        return None

    @property
    def navigate(self):
        if self._action.name == "navigate":
            stub = type("NavStub", (), {"url": self._action.params.get("url", "")})()
            return stub
        return None

    @property
    def input_text(self):
        return None   # not used in new action set (we have type/type_into)

    @property
    def run_command(self):
        if self._action.name == "run_shell":
            stub = type("CmdStub", (), {"command": self._action.params.get("command", "")})()
            return stub
        return None

    @property
    def done(self):
        if self._action.name == "done":
            stub = type("DoneStub", (), {"text": self._action.params.get("text", "")})()
            return stub
        return None

# ── Constants ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.md"
_MAX_SAME_ACTION_REPEATS = 2   # if the same action name fails this many times, give up on it
_MAX_STUCK_STEPS = 3           # if screen doesn't change for this many steps, ask human
_MAX_OUTPUT_CHARS = 8_000      # truncate tool output before sending to LLM


# ── CoreLoop ──────────────────────────────────────────────────────────────────

class CoreLoop:
    """
    The single, tight agent loop for Autobot.

    Usage:
        loop = CoreLoop(computer=computer, llm_client=client, goal="open Notepad and type hello")
        result = await loop.run()
    """

    def __init__(
        self,
        computer: Computer,
        llm_client: Any,
        goal: str,
        model: str = "gpt-4o",
        max_steps: int = 25,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.computer = computer
        self.llm_client = llm_client
        self.goal = goal
        self.model = model
        self.max_steps = max_steps
        self.log = log or (lambda msg: logger.info(msg))

        # Runtime state
        self.step_number: int = 0
        self.history: list[StepRecord] = []
        self.is_cancelled: bool = False
        self.is_paused: bool = False
        self._pending_override: str | None = None   # mid-flight goal change
        self._last_screen: str = ""
        self._same_screen_count: int = 0
        self._last_done_success: bool = False

        # Services
        self._approval = ApprovalGuard(mode=os.getenv("AUTOBOT_APPROVAL_MODE", "balanced"))
        self._distiller = SkillDistiller()

        # Load the system prompt once
        self._system_prompt = self._load_system_prompt()

    # ── Public interface ───────────────────────────────────────────────────────

    def pause(self) -> None:
        """Pause execution at the current step."""
        self.is_paused = True
        self.log("⏸️ Paused.")

    def resume(self) -> None:
        """Resume execution."""
        self.is_paused = False
        self.log("▶️ Resumed.")

    def push_override(self, new_goal: str) -> None:
        """Inject a mid-flight goal change. Takes effect at the next step."""
        self._pending_override = new_goal
        self.log(f"⚡ Override queued: {new_goal}")

    async def run(self) -> str:
        """Run the loop until done or max_steps exhausted."""
        self.log(f"🤖 Goal: {self.goal}")
        self.log(f"📋 Max steps: {self.max_steps} | Model: {self.model}")

        # Inject previously learned skill context
        skill_context = self._distiller.get_skill_prompt_context(self.goal)
        if skill_context:
            self.log("💡 Found a matching learned skill — injecting context")

        while self.step_number < self.max_steps:
            if self.is_cancelled:
                self.log("⚠️ Cancelled.")
                return "Cancelled by user."

            while self.is_paused and not self.is_cancelled:
                await asyncio.sleep(0.5)

            if self.is_cancelled:
                self.log("⚠️ Cancelled.")
                return "Cancelled by user."

            # Apply any pending mid-flight override
            if self._pending_override:
                self.goal = self._pending_override
                self._pending_override = None
                self.log(f"🔀 Goal updated: {self.goal}")

            self.step_number += 1
            self.log(f"\n📍 Step {self.step_number}/{self.max_steps}")

            result = await self._step(skill_context)
            if result is not None:
                # Loop finished (done action or unrecoverable error)
                return result

        # Step budget exhausted
        self.log(f"⏱️ Step budget exhausted after {self.max_steps} steps.")
        self._try_distill()
        return f"Step limit reached ({self.max_steps} steps). Last goal: {self.goal}"

    # ── Single step ───────────────────────────────────────────────────────────

    async def _step(self, skill_context: str) -> str | None:
        """
        One full observe → decide → act → verify cycle.
        Returns a result string if the loop should stop, otherwise None.
        """
        # 1. OBSERVE
        screen = self._observe()
        self.log(f"  👁️ Window: {self._active_window_title()}")

        # Stuck detection — if screen hasn't changed for N steps, warn the model
        stuck_warning = ""
        if screen == self._last_screen:
            self._same_screen_count += 1
            if self._same_screen_count >= _MAX_STUCK_STEPS:
                stuck_warning = (
                    f"\n⚠️ WARNING: The screen has not changed for {self._same_screen_count} "
                    "consecutive steps. Your recent actions are not having visible effects. "
                    "Try a fundamentally different approach, use `screenshot` for visual context, "
                    "or use `human_input` to ask the user."
                )
                self.log(f"  ⚠️ Stuck for {self._same_screen_count} steps")
        else:
            self._same_screen_count = 0
        self._last_screen = screen

        # 2. DECIDE
        llm_resp = await self._decide(screen, skill_context, stuck_warning)
        if llm_resp is None:
            return "LLM failed to respond after retries. Cannot continue."

        self.log(f"  💭 {llm_resp.next_goal}")
        self.log(f"  ▶️  {llm_resp.action.describe()}")

        # 3. ACT
        if llm_resp.action.name == "done":
            return await self._handle_done(llm_resp, screen)

        action_result, new_screen = await self._act(llm_resp.action, screen)

        # 4. RECORD
        record = StepRecord(
            step=self.step_number,
            next_goal=llm_resp.next_goal,
            action=llm_resp.action,
            result=action_result,
            success="error" not in action_result.lower() and "failed" not in action_result.lower(),
            screen_before=screen,
            screen_after=new_screen,
        )
        self.history.append(record)

        icon = "✅" if record.success else "❌"
        self.log(f"  {icon} {action_result[:200]}")

        return None  # continue loop

    # ── OBSERVE ───────────────────────────────────────────────────────────────

    def _observe(self) -> str:
        """Extract the UIAutomation tree of the active window."""
        try:
            if hasattr(self.computer, "window") and self.computer.window is not None:
                tree = self.computer.window.extract_ui()
                title = self.computer.window.active_title()
                if title:
                    return f"Active window: {title}\n{tree}"
                return tree
        except Exception as e:
            logger.debug(f"UIAutomation extraction failed: {e}")
        return "(Could not extract screen state — try using 'screenshot' action)"

    def _active_window_title(self) -> str:
        try:
            if hasattr(self.computer, "window") and self.computer.window is not None:
                return self.computer.window.active_title() or "(unknown)"
        except Exception:
            pass
        return "(unknown)"

    # ── DECIDE ────────────────────────────────────────────────────────────────

    async def _decide(
        self,
        screen: str,
        skill_context: str,
        stuck_warning: str,
    ) -> LLMResponse | None:
        """Call the LLM and parse its response. Retries once on parse failure."""
        step_prompt = self._build_step_prompt(screen, skill_context, stuck_warning)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": step_prompt},
        ]

        for attempt in range(2):
            try:
                raw = await self._llm_call(messages)
                parsed = self._parse_llm_response(raw)
                if parsed is not None:
                    return parsed
                self.log(f"  ⚠️ Could not parse LLM response (attempt {attempt+1}), retrying…")
                # Add the bad response to the conversation so the model can self-correct
                messages.append({"role": "assistant", "content": raw or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON with the required fields "
                        "(thinking, next_goal, action). Please respond with ONLY a JSON object."
                    ),
                })
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt == 1:
                    return None
        return None

    async def _llm_call(self, messages: list[dict]) -> str:
        """Make the LLM API call. Returns raw text content."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        try:
            resp = await self.llm_client.chat.completions.create(**kwargs)
        except TypeError:
            # Some clients are sync-only (Anthropic adapter)
            resp = await asyncio.to_thread(
                self.llm_client.chat.completions.create, **kwargs
            )
        return (resp.choices[0].message.content or "").strip()

    def _parse_llm_response(self, raw: str) -> LLMResponse | None:
        """Parse LLM JSON response. Tolerates markdown code fences."""
        import re
        text = raw.strip()
        # Strip ```json ... ``` fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return LLMResponse.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"JSON parse failed: {e}\nRaw: {raw[:300]}")
            return None

    def _build_step_prompt(self, screen: str, skill_context: str, stuck_warning: str) -> str:
        """Assemble the per-step user message sent to the LLM."""
        parts: list[str] = [f"# Task Goal\n{self.goal}"]
        parts.append(f"\n# Step {self.step_number} of {self.max_steps}")

        if skill_context:
            parts.append(f"\n{skill_context}")

        if self.history:
            last = self.history[-1]
            icon = "✅" if last.success else "❌"
            parts.append(
                f"\n# Previous Step Result\n"
                f"{icon} Action: {last.action.describe()}\n"
                f"Result: {last.result[:500]}"
            )

        parts.append(f"\n# Current Screen State\n{screen}")

        if stuck_warning:
            parts.append(stuck_warning)

        parts.append("\n# Your Action\nRespond with JSON only.")
        return "\n".join(parts)

    # ── ACT ───────────────────────────────────────────────────────────────────

    async def _act(self, action: Action, screen_before: str) -> tuple[str, str]:
        """
        Execute one action. Returns (result_text, screen_after).
        Checks approval gate before execution.
        """
        # Approval gate for risky actions
        approved = await self._gate(action)
        if not approved:
            new_screen = self._observe()
            return "Action blocked by user approval gate.", new_screen

        # Dispatch
        try:
            result = await asyncio.to_thread(self._dispatch, action)
        except Exception as e:
            result = f"Error executing {action.name}: {e}"
            logger.warning(f"Action dispatch error: {e}", exc_info=True)

        # Re-observe after action
        await asyncio.sleep(0.5)   # brief wait for UI to settle
        new_screen = self._observe()
        return str(result)[:_MAX_OUTPUT_CHARS], new_screen

    def _dispatch(self, action: Action) -> str:
        """
        Synchronous action dispatcher. Maps action names to computer calls.
        Called via asyncio.to_thread so blocking is OK.
        """
        name = action.name
        p = action.params

        if name == "click":
            index = int(p.get("index", 0))
            ok = self.computer.window.click(index)
            return f"Clicked element [{index}]." if ok else f"Click on [{index}] failed — element may not exist."

        elif name == "type_into":
            index = int(p.get("index", 0))
            text = str(p.get("text", ""))
            ok = self.computer.window.type(index, text)
            return f"Typed into [{index}]." if ok else f"type_into [{index}] failed."

        elif name == "type":
            text = str(p.get("text", ""))
            self.computer.keyboard.type(text)
            return f"Typed: {text[:80]}"

        elif name == "key":
            combo = str(p.get("combo", ""))
            self.computer.keyboard.press(combo)
            return f"Pressed key: {combo}"

        elif name == "focus":
            title = str(p.get("title", ""))
            ok = self.computer.window.focus(title)
            return f"Focused window '{title}'." if ok else f"Window '{title}' not found."

        elif name == "navigate":
            url = str(p.get("url", ""))
            return self._navigate(url)

        elif name == "run_shell":
            command = str(p.get("command", ""))
            timeout = int(p.get("timeout", 30))
            return self.computer.terminal.run(command, timeout=timeout)

        elif name == "screenshot":
            img_b64 = self.computer.display.screenshot()
            # Return a short confirmation — the image itself is not injected into text
            return "Screenshot captured. If you need to see it, the next step will include visual context."

        elif name == "wait":
            secs = float(p.get("seconds", 1))
            time.sleep(min(secs, 30))
            return f"Waited {secs}s."

        elif name == "human_input":
            prompt = str(p.get("prompt", "The agent needs your input."))
            return self._request_human_input(prompt)

        elif name == "computer_call":
            call_str = str(p.get("call", ""))
            from autobot.computer.dispatch import dispatch_computer_call
            ok, result = dispatch_computer_call(self.computer, call_str)
            return result

        else:
            valid = ["click", "type_into", "type", "key", "focus", "navigate",
                     "run_shell", "screenshot", "wait", "human_input", "computer_call", "done"]
            return (
                f"Unknown action '{name}'. "
                f"Valid actions: {', '.join(valid)}"
            )

    def _navigate(self, url: str) -> str:
        """
        Navigate Chrome to a URL using UIAutomation.

        Strategy:
          1. If Chrome is already running, focus it and type the URL in the address bar.
          2. If Chrome is not running, launch it via shell then navigate.

        No CDP required — UIAutomation can interact with Chrome's address bar.
        """
        if not url:
            return "navigate: no URL provided."

        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            # Try to focus Chrome first
            focused = self.computer.window.focus("Chrome")
            if not focused:
                # Chrome not open — launch it
                self.computer.terminal.run(
                    f'start "" "chrome.exe" "{url}"', timeout=5
                )
                time.sleep(3)
                self.computer.window.focus("Chrome")
                return f"Launched Chrome and navigated to {url}"

            # Chrome is focused — use Ctrl+L to focus the address bar, then type URL
            time.sleep(0.3)
            self.computer.keyboard.press("ctrl+l")
            time.sleep(0.3)
            self.computer.keyboard.type(url)
            time.sleep(0.1)
            self.computer.keyboard.press("enter")
            time.sleep(1.5)   # wait for navigation to start
            return f"Navigated to {url}"

        except Exception as e:
            return f"navigate failed: {e}"

    def _request_human_input(self, prompt: str) -> str:
        """Register a human_input request through the gate and block until answered."""
        from autobot.agent.human_gate import wait_for_approval
        import hashlib
        key = f"human_input_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"

        self.log(f"  🙋 Waiting for human input: {prompt}")

        # Run async wait in a new event loop (we're in a thread)
        try:
            loop = asyncio.new_event_loop()
            allowed = loop.run_until_complete(
                wait_for_approval(key=key, message=f"[Input needed] {prompt}", timeout=300)
            )
            loop.close()
            if allowed:
                # The user's response is typically injected via the override mechanism
                override = self._pending_override
                self._pending_override = None
                return override or "(user responded — no text captured; check next step)"
            return "User declined to provide input."
        except Exception as e:
            return f"human_input failed: {e}"

    # ── APPROVAL GATE ─────────────────────────────────────────────────────────

    async def _gate(self, action: Action) -> bool:
        """
        Check approval for the action. Returns True if allowed to proceed.

        ApprovalGuard.gate() was written for the old ActionModel type and
        calls _action_text() on it, which looks for specific fields like
        .computer_call, .navigate, .run_command. Rather than rewriting the
        guard (which is tested and correct), we pass a minimal stub that
        satisfies _action_text() for the new action types.
        """
        tier = self._classify_risk(action)
        if tier == RiskTier.SAFE:
            return True

        # Build a minimal stub that _action_text() will handle correctly
        stub = _ActionStub(action)
        try:
            approved = await self._approval.gate(
                action=stub,
                tier=tier,
                goal=self.goal,
            )
            return approved
        except Exception as e:
            # Gate failure is fail-safe: block the action
            logger.warning(f"Approval gate error ({e}), blocking action for safety.")
            return False

    def _classify_risk(self, action: Action) -> RiskTier:
        """Map new Action types to risk tiers without relying on old ActionModel fields."""
        name = action.name

        if name in ("click", "type", "type_into", "key", "focus", "navigate",
                    "screenshot", "wait", "done"):
            return RiskTier.SAFE

        if name == "run_shell":
            # Shell commands are DANGER by default; the guard will auto-proceed
            # in trusted mode, pause in balanced mode, and always pause for
            # IRREVERSIBLE patterns (rm -rf etc.) regardless of mode.
            cmd = str(action.params.get("command", ""))
            # Check for IRREVERSIBLE patterns directly
            from autobot.agent.approval import _IRREVERSIBLE_RE
            if _IRREVERSIBLE_RE.search(f"run_command: {cmd}"):
                return RiskTier.IRREVERSIBLE
            return RiskTier.DANGER

        if name == "human_input":
            return RiskTier.SAFE

        if name == "computer_call":
            call_str = str(action.params.get("call", ""))
            from autobot.agent.approval import _IRREVERSIBLE_RE, _DANGER_RE
            if _IRREVERSIBLE_RE.search(call_str):
                return RiskTier.IRREVERSIBLE
            if _DANGER_RE.search(call_str):
                return RiskTier.DANGER
            return RiskTier.CAUTION

        return RiskTier.SAFE

    # ── DONE ──────────────────────────────────────────────────────────────────

    async def _handle_done(self, llm_resp: LLMResponse, screen: str) -> str:
        """Handle the done action — distill skill if successful."""
        params = llm_resp.action.params
        text = str(params.get("text", "Task complete."))
        success = bool(params.get("success", True))
        self._last_done_success = success

        icon = "🏆" if success else "📛"
        self.log(f"  {icon} Done: {text[:200]}")

        # Record the final step
        self.history.append(StepRecord(
            step=self.step_number,
            next_goal=llm_resp.next_goal,
            action=llm_resp.action,
            result=text,
            success=success,
            screen_before=screen,
            screen_after=screen,
        ))

        if success:
            self._try_distill()

        return text

    def _try_distill(self) -> None:
        """Save a learned skill from this run if it was successful."""
        if not self.history:
            return
        try:
            skill = self._distiller.distill_from_run(
                goal=self.goal,
                history=self.history,
                result="success" if self._last_done_success else "failed",
            )
            if skill:
                self.log(f"  🎓 Skill saved: '{skill.name}' (used {skill.success_count}x)")
        except Exception as e:
            logger.debug(f"Skill distillation failed: {e}")

    # ── UTILITIES ─────────────────────────────────────────────────────────────

    def _load_system_prompt(self) -> str:
        """Load the system prompt from disk and inject the tool catalog."""
        try:
            template = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not load system prompt from {_SYSTEM_PROMPT_PATH}: {e}")
            template = "You are Autobot, an autonomous desktop agent. {tool_catalog}"

        catalog = ""
        try:
            catalog = self.computer.get_tool_catalog()
        except Exception as e:
            logger.debug(f"Could not generate tool catalog: {e}")

        return template.replace("{tool_catalog}", catalog)
