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
from typing import Any

from autobot.agent.approval import ApprovalGuard, RiskTier
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
    RequestHumanInputAction,
    RunCommandAction,
    ScrollAction,
    StepHistoryEntry,
)
from autobot.computer.computer import Computer
from autobot.dom.extraction import DOMExtractionService
from autobot.dom.models import BrowserState, DOMSerializedState
from autobot.knowledge.environment_memory import EnvironmentMemory
from autobot.knowledge.skill_distiller import SkillDistiller
from autobot.prompts.builder import StepPromptBuilder, SystemPromptBuilder

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The LLM failed repeatedly, so no further progress is possible.

    Raised to stop the run immediately rather than letting the agent spend its
    whole step budget on requests that cannot succeed.
    """


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
        max_steps: int = 25,
        max_actions_per_step: int = 5,
        use_vision: bool = True,
        custom_instructions: str | None = None,
        first_step_context: str | None = None,
    ):
        self.page = page
        self.llm_client = llm_client
        self.goal = goal
        self.model = model
        self.max_steps = max_steps
        self.max_actions_per_step = max_actions_per_step
        self.use_vision = use_vision
        self.custom_instructions = custom_instructions
        # Big-picture context (e.g. full mission plan) injected only into the
        # very first step's history slot — see MissionAgent, which gives each
        # objective a brief per-step reminder via custom_instructions and the
        # full mission picture once, here, so it doesn't eat context budget
        # every step.
        self.first_step_context = first_step_context

        # State
        self.step_number = 0
        self.history: list[StepHistoryEntry] = []
        self.previous_dom_state: DOMSerializedState | None = None
        # Whether the agent's done() reported success — gates skill distillation.
        self._last_done_success = False
        # If the LLM is unreachable (bad key, no credit, network down), every
        # step fails identically. Without a circuit breaker the agent silently
        # burns its ENTIRE step budget re-issuing a doomed request and then
        # reports "No steps were executed", which says nothing about the real
        # cause. Track consecutive failures and abort early with the actual error.
        self._consecutive_llm_failures = 0
        self._last_llm_error: str = ""
        self._max_consecutive_llm_failures = 3

        # ── Vision cost control ──────────────────────────────────────────
        # A screenshot is by far the most expensive part of a step (roughly
        # 1-2k tokens each, versus a few hundred for the DOM text). Most
        # browser steps don't need one: the DOM snapshot already names every
        # interactive element. So send vision only when it actually adds
        # information — see _should_use_vision().
        #   always    — every step (most expensive; use when debugging)
        #   auto      — first step, after a failure, or when the DOM is sparse
        #   never     — text only (cheapest; blind to canvas/image-only UIs)
        self.vision_mode = os.getenv("AUTOBOT_VISION_MODE", "auto").lower()
        if self.vision_mode not in ("always", "auto", "never"):
            logger.warning(f"Unknown AUTOBOT_VISION_MODE '{self.vision_mode}', using 'auto'")
            self.vision_mode = "auto"
        if not use_vision:
            self.vision_mode = "never"
        # Below this many interactive elements, the DOM probably failed to
        # describe the page (SPA still rendering, canvas app) and a screenshot
        # is worth its cost.
        self._sparse_dom_threshold = 5

        # Computer API for OS-level tools
        self.computer = Computer()
        self.env_memory = EnvironmentMemory()
        self.skill_distiller = SkillDistiller()
        self.approval_guard = ApprovalGuard()

        # Build system prompt
        tool_catalog = self.computer.get_tool_catalog()
        self.system_prompt_builder = SystemPromptBuilder(
            max_actions_per_step=max_actions_per_step,
            custom_instructions=custom_instructions,
            tool_catalog=tool_catalog,
        )
        self.system_prompt = self.system_prompt_builder.build()
        self.pending_override: str | None = None

    def push_override(self, new_instruction: str) -> None:
        """Mid-flight intervention: update goal or inject instruction from remote human input."""
        logger.info(f"🔄 Mid-flight override received: '{new_instruction}'")
        self.pending_override = new_instruction

    async def run(self) -> str:
        """
        Run the agent loop until completion or max_steps.

        Returns:
            The final result text from the done action, or a summary.
        """
        logger.info(f"🤖 Agent starting: '{self.goal}' (max {self.max_steps} steps)")

        while self.step_number < self.max_steps:
            try:
                result = await self._execute_step()

                if result is not None:
                    # Agent called "done" — task is complete
                    logger.info(f"✅ Agent finished at step {self.step_number + 1}: {result}")
                    self._distill_skill_if_successful(result)
                    return result

                self.step_number += 1

            except LLMUnavailableError:
                # Not recoverable by retrying — the model itself is unreachable.
                # Propagate so the user sees the real cause immediately instead
                # of a spun-out step budget.
                raise
            except Exception as e:
                logger.error(f"❌ Step {self.step_number + 1} failed: {e}")
                self.step_number += 1
                # Continue to next step — the agent can recover

        # Hit max steps without completing
        logger.warning(f"⚠️ Agent hit max steps ({self.max_steps}) without completing")
        return self._summarize_history()

    def _distill_skill_if_successful(self, result: str) -> None:
        """
        Save this run's working path as a reusable skill.

        This closes the loop that makes repeated work cheap: without it,
        get_skill_prompt_context() reads from a skills directory that nothing
        ever writes to, so every run re-derives from scratch at full token
        cost. Only genuine successes are recorded — an agent that gave up via
        done(success=False) has nothing worth teaching the next run.

        Never raises: failing to learn from a successful run must not turn it
        into a failed one.
        """
        if not self._last_done_success:
            logger.debug("Skipping skill distillation — run did not report success.")
            return
        try:
            skill = self.skill_distiller.distill_from_run(
                goal=self.goal, history=self.history, result=result
            )
            if skill:
                logger.info(
                    f"🎓 Distilled skill '{skill.name}' "
                    f"({len(skill.proven_steps)} proven steps, seen {skill.success_count}x)"
                )
        except Exception as e:
            logger.warning(f"Skill distillation failed (run still succeeded): {e}")

    async def _execute_step(self) -> str | None:
        """
        Execute one step of the agent loop.

        Returns:
            Result text if the agent called "done", None otherwise.
        """
        step_start = time.time()

        # Apply any mid-flight intervention overrides
        if self.pending_override:
            logger.info(f"⚡ Applying mid-flight override: {self.pending_override}")
            self.goal = f"{self.goal}\n\n[HUMAN INTERVENTION / RE-PIVOT]: {self.pending_override}"
            self.pending_override = None

        # ─── 1. OBSERVE ───
        logger.debug(f"Step {self.step_number + 1}: Observing...")
        dom_service = DOMExtractionService(
            self.page,
            previous_state=self.previous_dom_state,
            capture_screenshot=self.vision_mode != "never",
        )
        browser_state = await dom_service.extract_state()
        url_before = browser_state.url

        # Update previous state for new-element detection
        self.previous_dom_state = DOMSerializedState(
            element_tree=browser_state.element_tree,
            selector_map=browser_state.selector_map,
        )

        # If the user's foreground window is a native app rather than the
        # browser, the DOM snapshot above describes something the agent isn't
        # actually looking at. Capture the native window's UI tree too.
        native_context = await self._get_native_context()

        # ─── 2. THINK ───
        logger.debug(f"Step {self.step_number + 1}: Thinking...")
        agent_output = await self._call_llm(browser_state, native_context=native_context)

        if agent_output is None:
            self._consecutive_llm_failures += 1
            logger.error(
                f"LLM returned no output "
                f"({self._consecutive_llm_failures}/{self._max_consecutive_llm_failures} consecutive)"
            )
            if self._consecutive_llm_failures >= self._max_consecutive_llm_failures:
                raise LLMUnavailableError(
                    f"The LLM failed {self._consecutive_llm_failures} times in a row, so "
                    f"the agent cannot make progress. Last error:\n  {self._last_llm_error}\n\n"
                    "Common causes: no credit on the API account, an invalid or revoked "
                    "API key, no network access, or a model name that doesn't exist for "
                    "this provider. Run 'autobot --doctor' to check configuration."
                )
            return None

        self._consecutive_llm_failures = 0

        logger.info(
            f"Step {self.step_number + 1}: "
            f"Goal: {agent_output.next_goal} | "
            f"Actions: {len(agent_output.action)}"
        )

        # ─── 3. ACT ───
        logger.debug(f"Step {self.step_number + 1}: Acting...")
        action_results = await self._execute_actions(
            agent_output.action,
            browser_state,
        )

        # ─── 4. RECORD ───
        url_after = self._page_url()
        entry = StepHistoryEntry(
            step_number=self.step_number,
            agent_output=agent_output,
            action_results=action_results,
            url_before=url_before,
            url_after=url_after,
        )
        self.history.append(entry)

        step_time = time.time() - step_start
        logger.debug(f"Step {self.step_number + 1} completed in {step_time:.1f}s")

        # Check if agent called "done"
        for action in agent_output.action:
            if action.done is not None:
                self._last_done_success = action.done.success
                return action.done.text

        return None

    # Window-title fragments that mean "this is the browser" — for these the
    # DOM snapshot is the better description, so we skip native extraction
    # (which is slow, and would give the LLM a second competing index space).
    _BROWSER_TITLE_HINTS = ("chrome", "chromium", "edge", "firefox", "brave", "opera")

    async def _get_native_context(self) -> str:
        """
        Extract the focused native window's UI tree, if the focus is on a
        desktop app rather than the browser.

        Returns "" when native UI control is unavailable (non-Windows, or the
        optional `uiautomation` package isn't installed) or when the browser
        is focused. Never raises — perception failing must not kill the run.
        """
        window = getattr(self.computer, "window", None)
        if window is None:
            return ""

        try:
            title = await asyncio.to_thread(window.active_title)
            if not title:
                return ""
            if any(hint in title.lower() for hint in self._BROWSER_TITLE_HINTS):
                return ""

            tree = await asyncio.to_thread(window.extract_ui)
            if not tree:
                return ""

            logger.info(f"🖥️  Native window in focus: '{title}' — extracted UI tree")
            return (
                f"FOCUSED NATIVE WINDOW: {title}\n"
                "The browser state below does NOT describe this window. To act here use\n"
                "  computer.window.click(N) / computer.window.type(N, 'text')\n"
                "with the [N] indices from this tree (they are a SEPARATE index space\n"
                "from the browser's DOM indices — do not mix them up):\n"
                f"{tree}"
            )
        except Exception as e:
            logger.debug(f"Native window extraction skipped: {e}")
            return ""

    def _should_use_vision(self, browser_state: BrowserState) -> bool:
        """
        Decide whether this step is worth a screenshot.

        Sending an image every step is the single largest recurring cost in a
        run, and on a well-described DOM page it usually tells the model
        nothing the element list didn't already. We spend it where it pays:
        orienting on the first step, recovering after something went wrong,
        and whenever the text description looks too thin to act on.
        """
        if self.vision_mode == "never":
            return False
        if self.vision_mode == "always":
            return True

        # auto:
        if self.step_number == 0:
            return True  # orient once at the start

        # The DOM didn't describe much — likely a canvas app, an image-only
        # UI, or a page still rendering. Look at it directly.
        if browser_state.num_interactive < self._sparse_dom_threshold:
            return True

        # Something failed last step: the text state evidently wasn't enough
        # to choose a working action, so pay for eyes on the retry.
        if self.history:
            last = self.history[-1]
            if any(not r.success for r in last.action_results):
                return True

        return False

    async def _call_llm(
        self, browser_state: BrowserState, native_context: str = ""
    ) -> AgentOutput | None:
        """
        Call the LLM with the current state and parse the structured output.
        """
        # Build agent history text from previous steps.
        # At step 0 there's no history yet, so this is where a mission's
        # full-picture context (if any) gets its one-time injection.
        if self.step_number == 0 and self.first_step_context:
            history_text = self.first_step_context
        else:
            history_text = self._build_history_text()

        # Get environment knowledge and learned skill context
        env_summary = self.env_memory.get_summary_text()
        skill_context = self.skill_distiller.get_skill_prompt_context(self.goal)

        # Build the step prompt
        step_builder = StepPromptBuilder(
            browser_state=browser_state,
            task=self.goal,
            step_number=self.step_number,
            max_steps=self.max_steps,
            agent_history=history_text,
            environment_summary=env_summary,
            learned_skill_context=skill_context,
            native_window_context=native_context,
        )

        use_vision = self._should_use_vision(browser_state)
        if not use_vision:
            logger.debug(f"Step {self.step_number + 1}: text-only (vision skipped to save tokens)")
        user_messages = step_builder.build_messages(use_vision=use_vision)

        # Construct full message list
        messages = [
            {"role": "system", "content": self.system_prompt},
            *user_messages,
        ]

        try:
            response = await self._make_llm_call(messages)
            parsed = self._parse_agent_output(response)
            if parsed is None:
                self._last_llm_error = (
                    "the model replied, but its output was not valid JSON matching the "
                    "required action schema"
                )
            return parsed
        except Exception as e:
            self._last_llm_error = f"{type(e).__name__}: {e}"
            logger.error(f"LLM call failed: {self._last_llm_error}")
            return None

    async def _make_llm_call(self, messages: list[dict]) -> str:
        """Make the actual LLM API call. Supports both sync and async clients."""
        try:
            # Try async first
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except TypeError:
            # Fall back to sync client
            import asyncio
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

    def _parse_agent_output(self, raw: str) -> AgentOutput | None:
        """Parse the LLM's JSON response into an AgentOutput model."""
        try:
            # Handle markdown-wrapped JSON
            text = raw.strip()
            if text.startswith("```"):
                # Remove ```json ... ``` wrapper
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

            data = json.loads(text)
            return AgentOutput(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse LLM output: {e}\nRaw: {raw[:500]}")
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

            # Risk-gate before executing. SAFE actions (the vast majority —
            # click/navigate/scroll/etc.) return instantly with no overhead.
            # IRREVERSIBLE-tier actions always pause for a live human Allow,
            # in every approval mode including trusted — see approval.py.
            element_context = self._element_context_for_action(action, browser_state)
            tier = self.approval_guard.classify(action, element_context=element_context)
            if tier != RiskTier.SAFE:
                allowed = await self.approval_guard.gate(
                    action, tier, goal=self.goal, element_context=element_context,
                )
                if not allowed:
                    results.append(ActionResult(
                        action_name=action.action_name,
                        success=False,
                        error=f"Blocked by approval guard ({tier.value})",
                    ))
                    logger.warning(f"Action {i + 1} ({action.action_name}) blocked by approval guard — stopping this step")
                    break

            # Execute the action
            url_before = self._page_url()
            result = await self._execute_single_action(action, browser_state)
            results.append(result)

            # Page change detection (adapted from Browser Use)
            # If the page changed, skip remaining actions
            if result.page_changed or self._page_url() != url_before:
                remaining = len(actions) - i - 1
                if remaining > 0:
                    logger.info(
                        f"Page changed after action {i + 1}, "
                        f"skipping {remaining} remaining actions"
                    )
                break

            # Small delay between actions to mimic human behavior
            import asyncio
            await asyncio.sleep(0.3)

        return results

    async def _execute_computer_call(self, call_action: ComputerCallAction) -> ActionResult:
        """
        Execute an OS-level tool call from the injected tool catalog.

        This is the action that makes Autobot a computer-use agent rather than
        a browser-only agent: native window focus, mouse/keyboard on desktop
        apps, clipboard, filesystem, terminal. Parsing is structural and never
        eval()'d — see computer/dispatch.py. Risk gating already happened in
        _execute_actions before we got here.
        """
        from autobot.computer.dispatch import dispatch_computer_call

        call_str = call_action.call
        logger.info(f"🛠️  computer_call: {call_str}")
        success, result_text = await dispatch_computer_call(self.computer, call_str)

        return ActionResult(
            action_name="computer_call",
            success=success,
            extracted_content=result_text if success else None,
            error=None if success else result_text,
        )

    def _element_context_for_action(self, action: ActionModel, browser_state: BrowserState) -> str:
        """
        Build a string of the TARGET element's attributes for input_text actions,
        so ApprovalGuard can detect credential fields (type="password", a
        card-number label, etc.) from what's being filled in, not from the
        arbitrary text being typed into it.
        """
        if action.input_text is None:
            return ""
        element = browser_state.selector_map.get(action.input_text.index)
        if element is None:
            return ""
        attr_str = " ".join(f'{k}="{v}"' for k, v in element.attributes.items())
        return f"<{element.tag_name} {attr_str}> {element.text}".strip()

    def _page_url(self) -> str:
        """Current page URL, or "" when no browser is attached.

        AgentLoop can run without a browser (OS-only mode) — see
        AgentRunner.run — so every page access has to tolerate page=None.
        """
        if self.page is None:
            return ""
        try:
            return self._page_url()
        except Exception:
            return ""

    # Actions that cannot work without an attached browser.
    _BROWSER_ONLY_ACTIONS = (
        "navigate", "click", "input_text", "scroll_down", "scroll_up",
        "press_key", "switch_tab", "new_tab", "close_tab", "go_back",
    )

    async def _execute_single_action(
        self,
        action: ActionModel,
        browser_state: BrowserState,
    ) -> ActionResult:
        """
        Execute a single action on the browser or OS.

        Key innovation from Browser Use: click/input use DOM INDEX, not selectors.
        """
        action_name = action.action_name
        action_data = action.action_data

        if self.page is None and action_name in self._BROWSER_ONLY_ACTIONS:
            return ActionResult(
                action_name=action_name,
                success=False,
                error=(
                    f"'{action_name}' needs a browser, but none is attached "
                    "(OS-only mode). For desktop apps use computer_call instead, e.g. "
                    '{"computer_call": {"call": "computer.window.focus(\'Notepad\')"}} '
                    "then computer.window.extract_ui() to see its elements. "
                    "To get a browser, start Chrome with --remote-debugging-port=9222."
                ),
            )

        try:
            if action.navigate is not None:
                await self.page.goto(action.navigate.url, wait_until="domcontentloaded")
                return ActionResult(action_name="navigate", success=True, page_changed=True)

            elif action.click is not None:
                return await self._execute_click(action.click, browser_state)

            elif action.input_text is not None:
                return await self._execute_input(action.input_text, browser_state)

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
                for p in self.page.context.pages:
                    if str(hash(p))[-6:] == action.switch_tab.tab_id:
                        self.page = p
                        await p.bring_to_front()
                        return ActionResult(action_name="switch_tab", success=True, page_changed=True)
                return ActionResult(
                    action_name="switch_tab",
                    success=False,
                    error=f"Tab {action.switch_tab.tab_id} not found",
                )

            elif action.close_tab is not None:
                await self.page.close()
                pages = self.page.context.pages
                if pages:
                    self.page = pages[-1]
                return ActionResult(action_name="close_tab", success=True, page_changed=True)

            elif action.wait is not None:
                import asyncio
                await asyncio.sleep(action.wait.seconds)
                return ActionResult(action_name="wait", success=True)

            elif action.screenshot is not None:
                return ActionResult(action_name="screenshot", success=True)

            elif action.run_command is not None:
                return await self._execute_run_command(action.run_command)

            elif action.request_human_input is not None:
                logger.info(f"❓ Human input requested: '{action.request_human_input.prompt}'")
                return ActionResult(
                    action_name="request_human_input",
                    success=True,
                    extracted_content=f"Human input requested: {action.request_human_input.prompt}",
                )

            elif action.computer_call is not None:
                return await self._execute_computer_call(action.computer_call)

            else:
                # Tell the LLM exactly what it got wrong. Previously an
                # unrecognized action name produced a bare "Unknown action:
                # unknown", giving the model nothing to correct against — so
                # it would often emit the same bad action again next step.
                bad_keys = action.unrecognized_keys
                if bad_keys:
                    valid = ", ".join(n for n in action.model_fields)
                    error = (
                        f"Unrecognized action key(s): {', '.join(bad_keys)}. "
                        f"Valid actions are: {valid}. "
                        "To use an OS-level tool from the tool catalog, use "
                        '{"computer_call": {"call": "computer.<module>.<method>(...)"}}'
                    )
                else:
                    error = "Empty action — no action field was set."
                return ActionResult(action_name="unknown", success=False, error=error)

        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            return ActionResult(action_name=action_name, success=False, error=str(e))

    async def _execute_click(self, click: ClickAction, browser_state: BrowserState) -> ActionResult:
        """
        Click an element by its DOM index via CDP (computer.browser.click_element).

        The index comes from dom/extraction.py, which builds browser_state.selector_map
        from the SAME CDP snapshot (dom/page_snapshot.py) that click_element() re-queries
        by index — so [N] always means the same element in both places. click_element()
        re-reads the element's current bounding-rect right before dispatching real CDP
        mouse events, rather than trusting stale snapshot coordinates (DESIGN_PHILOSOPHY.md:
        "CDP DOM Query First").
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
            result_text = await asyncio.to_thread(
                self.computer.browser.click_element, index, browser_state.url
            )
        except Exception as e:
            result_text = f"error: {e}"

        if result_text.startswith("clicked "):
            logger.info(f"Clicked [{index}]: {result_text}")
            await asyncio.sleep(0.5)
            return ActionResult(
                action_name="click",
                success=True,
                page_changed=self._page_url() != browser_state.url,
            )

        # The CDP click didn't land. Rather than report failure and let the
        # model burn its next step re-issuing the identical click, escalate to
        # a physically different interaction — an element can be present in the
        # DOM but covered by an overlay, outside the viewport, or only
        # responsive to a real OS-level mouse event.
        return await self._click_fallback_ladder(index, element, browser_state, result_text)

    async def _click_fallback_ladder(
        self,
        index: int,
        element: Any,
        browser_state: BrowserState,
        first_error: str,
    ) -> ActionResult:
        """
        Try progressively different ways of clicking the same element.

        Each rung is a genuinely different mechanism, not a retry of the last
        one — repeating an identical failing action is the single most common
        way this agent used to waste its whole step budget.
        """
        attempts = [f"cdp_click: {first_error}"]

        # Rung 1: scroll it into view, then click again. Handles elements that
        # resolve fine but sit outside the current viewport.
        try:
            await asyncio.to_thread(self.computer.browser.scroll_to, index, browser_state.url)
            await asyncio.sleep(0.3)
            retry = await asyncio.to_thread(
                self.computer.browser.click_element, index, browser_state.url
            )
            if retry.startswith("clicked "):
                logger.info(f"Clicked [{index}] after scrolling into view")
                await asyncio.sleep(0.5)
                return ActionResult(
                    action_name="click", success=True,
                    page_changed=self._page_url() != browser_state.url,
                    extracted_content="recovered: needed scroll into view first",
                )
            attempts.append(f"scroll+cdp_click: {retry}")
        except Exception as e:
            attempts.append(f"scroll+cdp_click: {e}")

        # Rung 2: call the element's own .click() in JS. This dispatches no
        # mouse events at coordinates, so it bypasses pointer hit-testing —
        # the fix when the correct element is found but a transparent overlay,
        # cookie banner, or sticky header is intercepting the real click.
        try:
            js_result = await asyncio.to_thread(
                self.computer.browser.click_via_js, index, browser_state.url
            )
            if js_result.startswith("js-clicked "):
                logger.info(f"Clicked [{index}] via JS fallback (something was intercepting)")
                await asyncio.sleep(0.5)
                return ActionResult(
                    action_name="click", success=True,
                    page_changed=self._page_url() != browser_state.url,
                    extracted_content="recovered: normal click was intercepted; used JS click",
                )
            attempts.append(f"js_click: {js_result}")
        except Exception as e:
            attempts.append(f"js_click: {e}")

        # All rungs failed. Report every distinct thing that was tried, so the
        # model picks a different STRATEGY next step (different element, scroll,
        # dismiss an overlay) instead of the same click a third time.
        return ActionResult(
            action_name="click",
            success=False,
            error=(
                f"Click on [{index}] <{element.tag_name}> '{element.text[:40]}' failed "
                f"after {len(attempts)} different methods:\n  - " + "\n  - ".join(attempts)
                + "\nDo NOT retry this same click. The element may be covered by an "
                "overlay/modal, disabled, or inside an iframe. Try dismissing any "
                "popup, scrolling, or targeting a different element."
            ),
        )

    async def _execute_input(self, input_action: InputTextAction, browser_state: BrowserState) -> ActionResult:
        """
        Type text into an element by its DOM index via CDP (computer.browser.fill).
        Same index space and same reasoning as _execute_click above. fill() clears
        the field, inserts text via CDP Input.insertText, then re-reads the field
        to verify the text actually landed — we treat unverified fills as failures
        so the agent retries instead of assuming success on a field that silently
        rejected input (a common failure mode on rich-text editors like Grok/ChatGPT).
        """
        index = input_action.index
        if browser_state.selector_map.get(index) is None:
            return ActionResult(
                action_name="input_text",
                success=False,
                error=f"Element with index {index} not found in selector map",
            )

        try:
            result_text = await asyncio.to_thread(
                self.computer.browser.fill, index, input_action.text, browser_state.url
            )
        except Exception as e:
            return ActionResult(action_name="input_text", success=False, error=f"Input to [{index}] failed: {e}")

        success = result_text.startswith("filled ") and "verified: True" in result_text
        logger.info(f"Input [{index}]: {result_text}")

        return ActionResult(
            action_name="input_text",
            success=success,
            error=None if success else result_text,
        )

    def _build_history_text(self) -> str:
        """Build a text summary of all previous steps for the agent_history section."""
        if not self.history:
            return ""
            
        lines = []
        for entry in self.history[-5:]:  # Last 5 steps to keep context manageable
            lines.append(entry.to_history_text())
            
        # --- Stall / Loop Detection ---
        if len(self.history) >= 3:
            last_3 = self.history[-3:]
            # Only compare the list of actions (dumping to JSON to easily match dicts)
            acts_0 = [a.model_dump() for a in last_3[0].agent_output.action]
            acts_1 = [a.model_dump() for a in last_3[1].agent_output.action]
            acts_2 = [a.model_dump() for a in last_3[2].agent_output.action]
            
            if acts_0 == acts_1 == acts_2:
                logger.warning(f"🔄 Loop detected: Agent repeated the exact same actions for 3 steps.")
                lines.append(
                    "\n> [!CRITICAL WARNING]\n"
                    "> ⚠️ STALL DETECTED ⚠️\n"
                    "> You have executed the EXACT SAME actions for the last 3 steps and made no progress.\n"
                    "> Do NOT repeat these actions. You MUST try a completely different strategy,\n"
                    "> interact with different elements, scroll to find new elements, or use the `done` tool."
                )
                
        return "\n".join(lines)

    def _summarize_history(self) -> str:
        """Generate a summary when the agent hits max steps."""
        if not self.history:
            return "No steps were executed."

        steps_text = [entry.to_history_text() for entry in self.history[-3:]]
        return (
            f"Agent ran {len(self.history)} steps without calling 'done'.\n"
            f"Last steps:\n" + "\n".join(steps_text)
        )

    async def _execute_run_command(self, cmd_action: RunCommandAction) -> ActionResult:
        """Execute a local shell command safely."""
        import asyncio
        from pathlib import Path

        cmd = cmd_action.command
        logger.info(f"💻 Running local command: {cmd}")

        scratch_dir = Path.cwd() / "tmp" / "autobot_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(scratch_dir),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=float(cmd_action.timeout)
                )
            except asyncio.TimeoutError:
                process.kill()
                return ActionResult(
                    action_name="run_command",
                    success=False,
                    error=f"Command timed out after {cmd_action.timeout}s: {cmd}",
                )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()
            output = f"STDOUT:\n{stdout_str}\n\nSTDERR:\n{stderr_str}" if stderr_str else stdout_str

            return ActionResult(
                action_name="run_command",
                success=process.returncode == 0,
                extracted_content=output[:2000],
                error=f"Exit code {process.returncode}" if process.returncode != 0 else None,
            )
        except Exception as e:
            return ActionResult(
                action_name="run_command",
                success=False,
                error=f"Failed to execute command: {e}",
            )
