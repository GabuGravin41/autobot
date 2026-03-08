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

import json
import logging
import time
from typing import Any

from autobot.agent.models import (
    ActionModel,
    ActionResult,
    AgentOutput,
    AgentStepInfo,
    ClickAction,
    DoneAction,
    InputTextAction,
    NavigateAction,
    PressKeyAction,
    ScrollAction,
    StepHistoryEntry,
)
from autobot.computer.computer import Computer
from autobot.dom.extraction import DOMExtractionService
from autobot.dom.models import BrowserState, DOMSerializedState
from autobot.prompts.builder import StepPromptBuilder, SystemPromptBuilder

logger = logging.getLogger(__name__)


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
    ):
        self.page = page
        self.llm_client = llm_client
        self.goal = goal
        self.model = model
        self.max_steps = max_steps
        self.max_actions_per_step = max_actions_per_step
        self.use_vision = use_vision
        self.custom_instructions = custom_instructions

        # State
        self.step_number = 0
        self.history: list[StepHistoryEntry] = []
        self.previous_dom_state: DOMSerializedState | None = None

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

    def get_status(self) -> dict[str, Any]:
        """Returns the current status metadata for the dashboard."""
        return {
            "current_step": self.step_number,
            "max_steps": self.max_steps,
            "goal": self.goal,
            "last_screenshot_path": self.last_screenshot_path,
        }

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
                    return result

                self.step_number += 1

            except Exception as e:
                logger.error(f"❌ Step {self.step_number + 1} failed: {e}")
                self.step_number += 1
                # Continue to next step — the agent can recover

        # Hit max steps without completing
        logger.warning(f"⚠️ Agent hit max steps ({self.max_steps}) without completing")
        return self._summarize_history()

    async def _execute_step(self) -> str | None:
        """
        Execute one step of the agent loop.

        Returns:
            Result text if the agent called "done", None otherwise.
        """
        step_start = time.time()

        # ─── 1. OBSERVE ───
        logger.debug(f"Step {self.step_number + 1}: Observing...")
        dom_service = DOMExtractionService(
            self.page,
            previous_state=self.previous_dom_state,
        )
        browser_state = await dom_service.extract_state()
        url_before = browser_state.url

        # Save screenshot for live-view (Mini-Peek)
        if browser_state.screenshot_b64:
            try:
                import base64
                from pathlib import Path
                screenshot_dir = Path("screenshots")
                screenshot_dir.mkdir(exist_ok=True)
                screenshot_path = screenshot_dir / f"latest.png"
                screenshot_path.write_bytes(base64.b64decode(browser_state.screenshot_b64))
                self.last_screenshot_path = str(screenshot_path.absolute())
            except Exception as e:
                logger.warning(f"Failed to save agent screenshot: {e}")

        # Update previous state for new-element detection
        self.previous_dom_state = DOMSerializedState(
            element_tree=browser_state.element_tree,
            selector_map=browser_state.selector_map,
        )

        # ─── 2. THINK ───
        logger.debug(f"Step {self.step_number + 1}: Thinking...")
        agent_output = await self._call_llm(browser_state)

        if agent_output is None:
            logger.error("LLM returned no output")
            return None

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
        url_after = self.page.url
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
                return action.done.text

        return None

    async def _call_llm(self, browser_state: BrowserState) -> AgentOutput | None:
        """
        Call the LLM with the current state and parse the structured output.
        """
        # Build agent history text from previous steps
        history_text = self._build_history_text()

        # Build the step prompt
        step_builder = StepPromptBuilder(
            browser_state=browser_state,
            task=self.goal,
            step_number=self.step_number,
            max_steps=self.max_steps,
            agent_history=history_text,
        )

        user_messages = step_builder.build_messages(use_vision=self.use_vision)

        # Construct full message list
        messages = [
            {"role": "system", "content": self.system_prompt},
            *user_messages,
        ]

        try:
            response = await self._make_llm_call(messages)
            return self._parse_agent_output(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
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

            # Execute the action
            url_before = self.page.url
            result = await self._execute_single_action(action, browser_state)
            results.append(result)

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
            import asyncio
            await asyncio.sleep(0.3)

        return results

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

            elif action.computer_call is not None:
                return await self._execute_computer_call(action.computer_call)

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
            import asyncio
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

    async def _execute_computer_call(self, call_action: ComputerCallAction) -> ActionResult:
        """
        Execute an OS-level computer tool call.
        Example: "computer.mouse.click(x=10, y=10)"
        """
        call_str = call_action.call
        logger.info(f"💻 Computer call: {call_str}")

        try:
            # We use a restricted eval-like approach by accessing the computer object
            # Matches strings like "computer.mouse.click(...)"
            if not call_str.startswith("computer."):
                return ActionResult(
                    action_name="computer_call",
                    success=False,
                    error=f"Invalid computer call: {call_str}. Must start with 'computer.'"
                )

            # Resolve the method
            parts = call_str.split("(", 1)[0].split(".")
            target_obj = self.computer
            for part in parts[1:]:
                target_obj = getattr(target_obj, part)

            # Parse arguments (basic implementation, similar to Open Interpreter)
            args_str = call_str.split("(", 1)[1].rstrip(")")
            # Using a safer way to evaluate arguments if possible, or just passing them
            # For now, we'll use a very simple parser or just use eval in a controlled way
            # since the LLM is expected to provide valid Python-like calls.
            
            def safe_eval_call(target, args_text):
                # This is a bit risky but we are in a sovereign agent context
                # and the LLM is controlled.
                import ast
                # We could use a proper parser but for speed/simplicity:
                return target(*eval(f"({args_text})", {"__builtins__": {}}, {}))

            if args_str.strip():
                # We wrap the call in a lambda to handle *args and **kwargs via eval
                # This is a placeholder for a more robust parser
                result = await asyncio.to_thread(
                    lambda: eval(call_str, {"computer": self.computer, "self": self}, {})
                )
            else:
                result = await asyncio.to_thread(target_obj)

            return ActionResult(
                action_name="computer_call",
                success=True,
                extracted_content=str(result) if result is not None else None
            )

        except Exception as e:
            logger.error(f"Computer call failed: {e}")
            return ActionResult(
                action_name="computer_call",
                success=False,
                error=f"Computer tool failed: {e}. Fallback: If this API/tool is unavailable, navigate to the website in the browser and complete the task manually using the Human Profile."
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

# Alias for compatibility with the backend (app.py)
AgentRunner = AgentLoop
