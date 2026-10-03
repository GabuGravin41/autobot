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
  - For actual web page DOM content (which UIAutomation sees unreliably —
    Chrome's toolbar/tabs, not consistently the page itself), the
    browser_text / browser_list / browser_click / browser_type actions go
    through the Chrome extension's content script (see
    autobot/browser/extension_bridge.py) instead of CDP — no debug port,
    no profile launch, no SingletonLock fights.
  - Kaggle (kernel pull/push/status/output — autobot/computer/kaggle_tool.py)
    and Claude Code (headless runs — autobot/computer/claude_code_tool.py)
    are reached through the existing `computer_call` action, i.e.
    `computer.kaggle.pull_kernel(...)` / `computer.claude_code.run_headless(...)`
    — real APIs/CLIs, not UI automation of either tool's web/chat interface.
    No new action names needed: computer_call's AST-safe dispatcher
    (autobot/computer/dispatch.py) already reaches any public method on any
    Computer submodule, and _classify_risk() below already risk-classifies
    computer_call by pattern-matching the call string.
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

from autobot.agent.action_models import ACTION_PARAMS, Action, LLMResponse, StepRecord
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
_HISTORY_WINDOW = 8            # recent steps shown to the model every step
_MAX_IDENTICAL_REPEATS = 3     # identical action+result this many times -> blocked
_PARSE_ATTEMPTS = 3            # re-asks when the model's reply isn't a valid action


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
        self._pending_image_b64: str | None = None   # screenshot to show on the next step
        self._vision = os.getenv("AUTOBOT_VISION", "").strip().lower() in ("1", "true", "yes", "on")

        # Services
        # ApprovalGuard also reads AUTOBOT_UNATTENDED itself (unattended=None
        # here means "check the env var") — set it before starting a run
        # you're stepping away from (e.g. Kaggle kernel iteration overnight)
        # so CAUTION/DANGER don't stall waiting for an Allow click nobody's
        # there to give. See approval.py's module docstring for the exact
        # behavior this changes, and note the IRREVERSIBLE floor doesn't move.
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

        # A cancel that arrived while the LLM call was in flight must not be
        # followed by one more real action (the Sep 7 run log shows exactly
        # that: "Task cancelled", then another computer_call).
        if self.is_cancelled:
            self.log("⚠️ Cancelled.")
            return "Cancelled by user."

        # 3. ACT
        if llm_resp.action.name == "done":
            return await self._handle_done(llm_resp, screen)

        repeats = self._identical_repeats(llm_resp.action)
        if repeats >= _MAX_IDENTICAL_REPEATS:
            # The model has already run this exact action this many times and
            # got the same result each time. Running it again cannot help; the
            # Sep 7 Overleaf run spent 12 steps cycling through three calls
            # like this. Refuse, and say so plainly in the next prompt.
            action_result = (
                f"BLOCKED: you have already run {llm_resp.action.describe()} {repeats} times "
                f"and it returned the same result every time. It will not be run again. "
                f"Choose a different action, or use human_input, or finish with done(success=false)."
            )
            new_screen = screen
        else:
            action_result, new_screen = await self._act(llm_resp.action, screen)

        # 4. RECORD
        record = StepRecord(
            step=self.step_number,
            next_goal=llm_resp.next_goal,
            action=llm_resp.action,
            result=action_result,
            success=self._looks_successful(action_result),
            screen_before=screen,
            screen_after=new_screen,
        )
        self.history.append(record)

        icon = "✅" if record.success else "❌"
        self.log(f"  {icon} {action_result[:200]}")

        return None  # continue loop

    # ── OBSERVE ───────────────────────────────────────────────────────────────

    # Window titles that mean "this is Chrome" — used to decide whether to
    # append the perception hint below. Substring match, case-insensitive,
    # same spirit as window.py's focus() fix.
    _BROWSER_TITLE_MARKERS = ("chrome",)

    def _observe(self) -> str:
        """Extract the current screen state.

        UIAutomation (the active window's element tree) is the base layer
        for every window, browser included. But when the active window IS
        Chrome, that tree only ever shows the toolbar/tabs — never the page
        itself (see system_prompt.md's Browser Notes) — so a model relying
        on OBSERVE alone would see an empty-looking browser every single
        step and have no signal that a second perception source
        (browser_text/browser_list, via the extension bridge) exists and is
        what it actually needs here. Previously the model had to *guess*
        that on its own from static system-prompt instructions alone, with
        no per-step reminder — this appends an explicit, cheap (no network
        call — just a string check on the window title already read this
        step) hint instead of leaving that judgment call to the model each
        time. This is a hint, not an eager fetch: it does not itself call
        browser_text, so it costs nothing beyond the one string comparison
        and doesn't add a bridge round-trip on steps where the model was
        about to do something else entirely (e.g. click Chrome's own
        toolbar).
        """
        title = self._active_window_title()
        try:
            if hasattr(self.computer, "window") and self.computer.window is not None:
                tree = self.computer.window.extract_ui()
                screen = f"Active window: {title}\n{tree}" if title and title != "(unknown)" else tree
            else:
                screen = "(Could not extract screen state — try using 'screenshot' action)"
        except Exception as e:
            logger.debug(f"UIAutomation extraction failed: {e}")
            screen = "(Could not extract screen state — try using 'screenshot' action)"

        if self._looks_like_browser(title):
            screen += (
                "\n\n[Perception hint: the active window is Chrome. The tree above is "
                "UIAutomation's view of Chrome's own toolbar/tabs — it will NOT show the "
                "web page's content, links, or text. Use browser_text or browser_list now "
                "to see what's actually on the page before deciding your next action.]"
            )
        return screen

    def _looks_like_browser(self, title: str) -> bool:
        t = (title or "").lower()
        return any(marker in t for marker in self._BROWSER_TITLE_MARKERS)

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
        user_content: Any = step_prompt
        if self._pending_image_b64 and self._vision:
            user_content = [
                {"type": "text", "text": step_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._pending_image_b64}"}},
            ]
        self._pending_image_b64 = None
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]

        for attempt in range(_PARSE_ATTEMPTS):
            try:
                raw = await self._llm_call(messages)
                parsed = self._parse_llm_response(raw)
                if parsed is not None:
                    return parsed
                self.log(f"  ⚠️ Could not parse a valid action (attempt {attempt+1}/{_PARSE_ATTEMPTS}), re-asking…")
                messages.append({"role": "assistant", "content": raw or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "That reply did not contain a usable action. Reply with ONE JSON object exactly like:\n"
                        '{"thinking": "...", "next_goal": "...", "action": {"name": "click", "params": {"index": 3}}}\n'
                        f"Valid action names: {', '.join(sorted(ACTION_PARAMS))}."
                    ),
                })
            except Exception as e:
                # Surface this to the run log (self.log), not just Python's
                # logging module — logger.error() alone lands in the
                # backend's OWN terminal (python -m autobot.main), which is
                # a different window than wherever the run's log is being
                # watched (dashboard, /api/logs, this script). Without this,
                # every real cause (bad key, rate limit, unknown model,
                # network error) collapses into the same opaque "LLM failed
                # to respond after retries" with no way to tell them apart.
                msg = f"LLM call failed (attempt {attempt+1}/{_PARSE_ATTEMPTS}): {type(e).__name__}: {e}"
                logger.error(msg)
                self.log(f"  ⚠️ {msg}")
                if attempt >= 1:
                    return None
        return None

    async def _llm_call(self, messages: list[dict]) -> str:
        """
        Make the LLM API call. Returns raw text content.

        self.llm_client is usually a SYNCHRONOUS client (openai.OpenAI(...),
        or the Anthropic adapter — both real network calls under a plain
        `def create(...)`, not `async def`). Calling create() already
        performs the request; only the subsequent `await` on its non-
        awaitable result would fail. So: check whether create is actually a
        coroutine function BEFORE calling it, and route accordingly — never
        call it once to find out, throw that (already-executed, already-
        billed) response away, and call it again. The previous version did
        exactly that on every single step for a sync client, which is the
        common case here — double cost, double latency, and if a provider
        rate-limits two rapid calls, the second (real, kept) attempt could
        fail because of the first (wasted, discarded) one.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        create = self.llm_client.chat.completions.create
        if asyncio.iscoroutinefunction(create):
            resp = await create(**kwargs)
        else:
            resp = await asyncio.to_thread(create, **kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _parse_llm_response(self, raw: str) -> LLMResponse | None:
        """Parse the model's reply into a real action, or None (never a default action)."""
        parsed = LLMResponse.parse(raw)
        if parsed is None:
            logger.debug(f"Unparseable model reply: {str(raw)[:300]}")
        return parsed

    # ── History helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _looks_successful(result: str) -> bool:
        r = (result or "").strip().lower()
        if not r or r in ("none", "[]", "{}", '""', "null"):
            return False   # an empty result is not evidence that anything worked
        return not any(w in r for w in ("error", "failed", "blocked:", "not found", "traceback"))

    @staticmethod
    def _signature(action: Action) -> str:
        return json.dumps([action.name, action.params], sort_keys=True, default=str)

    def _identical_repeats(self, action: Action) -> int:
        """How many times this exact action already ran AND produced the same result."""
        sig = self._signature(action)
        # BLOCKED notices don't count as a new result — otherwise a block would
        # reset the counter and let the very next identical call through.
        results = [h.result for h in self.history
                   if self._signature(h.action) == sig and not h.result.startswith("BLOCKED:")]
        if not results:
            return 0
        last = results[-1]
        return sum(1 for r in results if r == last)

    def _repeated_summary(self) -> str:
        """Actions run 2+ times with an identical result, one line each."""
        seen: dict[str, tuple[Action, str, int]] = {}
        for h in self.history:
            key = self._signature(h.action) + "\x00" + h.result
            a, r, n = seen.get(key, (h.action, h.result, 0))
            seen[key] = (a, r, n + 1)
        lines = [
            f"  - {a.describe()[:120]} (x{n}) -> {' '.join(r.split())[:100] or '(empty result)'}"
            for a, r, n in seen.values() if n >= 2
        ]
        return "\n".join(lines)

    def _build_step_prompt(self, screen: str, skill_context: str, stuck_warning: str) -> str:
        """Assemble the per-step user message sent to the LLM."""
        parts: list[str] = [f"# Task Goal\n{self.goal}"]
        parts.append(f"\n# Step {self.step_number} of {self.max_steps}")

        if skill_context:
            parts.append(f"\n{skill_context}")

        if self.history:
            # The model used to see ONLY the last step, so it could not know it
            # had already tried something three steps ago — that is how the
            # Sep 7 run cycled through the same three calls for 12 steps.
            recent = self.history[-_HISTORY_WINDOW:]
            lines = []
            for h in recent:
                icon = "OK " if h.success else "ERR"
                result = " ".join(h.result.split())[:160]
                lines.append(f"  step {h.step} [{icon}] {h.action.describe()[:120]} -> {result}")
            skipped = len(self.history) - len(recent)
            header = f"\n# Steps So Far ({len(self.history)} total"
            header += f", oldest {skipped} not shown)" if skipped else ")"
            parts.append(header + "\n" + "\n".join(lines))

            last = self.history[-1]
            parts.append(f"\n# Previous Step Full Result\n{last.result[:1500]}")

            repeated = self._repeated_summary()
            if repeated:
                parts.append(
                    "\n# Already Tried (same result each time — do NOT repeat these)\n" + repeated
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
            if not self._vision:
                # Previously this claimed "the next step will include visual
                # context" — it never did. Say what is actually true.
                return (
                    "Screenshot captured, but vision is OFF for this model (AUTOBOT_VISION is not set), "
                    "so you cannot see it. Use the element tree, browser_text/browser_list, or human_input instead."
                )
            if isinstance(img_b64, str) and len(img_b64) > 100:
                self._pending_image_b64 = img_b64
                return "Screenshot captured. It is attached to your next step."
            return f"Screenshot failed: {str(img_b64)[:200]}"

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
            # dispatch_computer_call is `async def`. This method (_dispatch)
            # is itself synchronous — it's invoked via asyncio.to_thread from
            # _act() specifically so blocking calls are safe here — which
            # means there is no event loop already running on this thread.
            # Calling the coroutine function without awaiting it does NOT
            # run its body; it just constructs a coroutine object, and the
            # `ok, result = <that object>` below used to raise "cannot
            # unpack non-iterable coroutine object" immediately. That
            # exception propagated out of _dispatch() and was caught by
            # _act()'s try/except as a generic "Error executing
            # computer_call: ..." — so every computer_call action was
            # silently failing, every time. Since computer.kaggle.*,
            # computer.research.*, computer.vault.*, and computer.files.*
            # have no dedicated action names of their own (unlike
            # run_shell/navigate/browser_*), computer_call is the ONLY way
            # to reach them — so all four tool modules were unreachable
            # from any real agent run until this fix. asyncio.run() here is
            # safe precisely because this thread has no other loop to
            # conflict with.
            ok, result = asyncio.run(dispatch_computer_call(self.computer, call_str))
            return result

        elif name == "browser_text":
            return self._browser_bridge_call("read_text")

        elif name == "browser_list":
            return self._browser_bridge_call("list_elements")

        elif name == "browser_click":
            index = int(p.get("index", 0))
            return self._browser_bridge_call("click_index", index=index)

        elif name == "browser_type":
            index = int(p.get("index", 0))
            text = str(p.get("text", ""))
            return self._browser_bridge_call("type_index", index=index, text=text)

        elif name == "browser_paste":
            # For CodeMirror/Monaco/ProseMirror-style rich editors (e.g.
            # Overleaf's LaTeX editor) where browser_type's direct
            # value/textContent write doesn't reliably register — see
            # extension/content.js's paste_text handler and its module
            # docstring for why a synthetic paste event is used instead.
            index = int(p.get("index", 0))
            text = str(p.get("text", ""))
            return self._browser_bridge_call("paste_text", index=index, text=text)

        else:
            valid = ["click", "type_into", "type", "key", "focus", "navigate",
                     "run_shell", "screenshot", "wait", "human_input", "computer_call",
                     "browser_text", "browser_list", "browser_click", "browser_type",
                     "browser_paste", "done"]
            return (
                f"Unknown action '{name}'. "
                f"Valid actions: {', '.join(valid)}"
            )

    def _browser_bridge_call(self, cmd_type: str, **params: Any) -> str:
        """
        Call into the Chrome extension DOM bridge (autobot/browser/
        extension_bridge.py) and return a compact text result for the LLM.

        This is the CDP-free path for actual web page content: UIAutomation
        sees Chrome's toolbar/tabs but not reliably the page DOM (see the
        "Browser Notes" section of system_prompt.md), and CDP requires
        launching/attaching to a debug port, which is exactly the profile-
        lock fragility this project moved away from. The already-installed
        Autobot extension talks to the page directly through its content
        script instead.
        """
        from autobot.browser.extension_bridge import bridge
        result = bridge.run(cmd_type, timeout=10.0, **params)
        if not result.get("ok"):
            return f"{cmd_type} failed: {result.get('error', 'unknown error')}"
        return json.dumps(result.get("data"))[:_MAX_OUTPUT_CHARS]

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
                    "screenshot", "wait", "done",
                    "browser_text", "browser_list", "browser_click", "browser_type",
                    "browser_paste"):
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
            from autobot.agent.approval import _IRREVERSIBLE_RE, _DANGER_RE, _SAFE_COMPUTER_CALL_RE
            if _IRREVERSIBLE_RE.search(call_str):
                return RiskTier.IRREVERSIBLE
            if _DANGER_RE.search(call_str):
                return RiskTier.DANGER
            if _SAFE_COMPUTER_CALL_RE.search(call_str):
                # Kaggle kernel pull/push/status/output — create/edit/read a
                # notebook in the user's own account. Explicitly SAFE (never
                # pauses, in any approval mode) so iteration stays frictionless;
                # only a real competition submit() is hard-gated above.
                return RiskTier.SAFE
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
