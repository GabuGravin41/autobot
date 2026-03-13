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
import time
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
from autobot.computer.computer import Computer
from autobot.dom.models import (
    BrowserState,
    DOMSerializedState,
    SelectorMap,
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

        # ─── 1. OBSERVE (Vision-Only) ───
        logger.debug(f"Step {self.step_number + 1}: Capturing vision observation...")
        
        import base64
        screenshot_bytes = await self.page.screenshot()

        # Get screen resolution for coordinate guidance in prompt
        try:
            screen_w, screen_h = self.computer.display.size()
        except Exception:
            screen_w, screen_h = 1920, 1080

        # Save full-res screenshot for dashboard live-view BEFORE downscaling
        try:
            from pathlib import Path
            screenshot_dir = Path("screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            screenshot_path = screenshot_dir / "latest.png"
            screenshot_path.write_bytes(screenshot_bytes)
            self.last_screenshot_path = str(screenshot_path.absolute())
        except Exception as e:
            logger.warning(f"Failed to save agent screenshot: {e}")

        # Compress screenshot PNG → JPEG to save tokens/money (no resolution change!)
        llm_screenshot_b64 = self._compress_screenshot(screenshot_bytes)

        browser_state = BrowserState(
            url=self.page.url,
            title=f"Human Mode | Screen: {screen_w}×{screen_h}",
            tabs=[],
            screenshot_b64=llm_screenshot_b64,
            element_tree=None,
            selector_map=SelectorMap(),
            num_links=0,
            num_interactive=0,
            total_elements=0,
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

        # ─── 2. THINK ───
        logger.debug(f"Step {self.step_number + 1}: Thinking...")
        
        # Build the step prompt
        step_builder = StepPromptBuilder(
            browser_state=browser_state,
            task=self.goal,
            step_number=self.step_number,
            max_steps=self.max_steps,
            agent_history=self._build_history_text(),
            native_ui=native_ui,
        )

        user_messages = step_builder.build_messages(use_vision=self.use_vision)

        # Construct full message list
        messages = [
            {"role": "system", "content": self.system_prompt},
            *user_messages,
        ]

        agent_output = await self._call_llm(messages)

        if agent_output is None:
            # LLM failed or refused. 
            # Stability: Wait before retrying (prevents spinning)
            logger.error(f"Step {self.step_number + 1}: LLM returned no output or invalid JSON. Waiting 3s...")
            await asyncio.sleep(3.0) 
            
            # Record a "Failed" entry in history so the user sees it
            fallback_output = AgentOutput(
                thinking="LLM call failed (possibly due to API limits or model error). Retrying...",
                next_goal="Retry current step",
                action=[]
            )
            entry = StepHistoryEntry(
                step_number=self.step_number,
                agent_output=fallback_output,
                action_results=[ActionResult(action_name="llm_call", success=False, error="API returned no valid response")],
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

    async def _call_llm(self, messages: list[dict]) -> AgentOutput | None:
        """
        Call the LLM with the current state and parse the structured output.
        _make_llm_call handles format/vision fallbacks internally.
        This layer adds one outer retry for transient network errors.
        """
        for attempt in range(1, 3):
            try:
                response = await self._make_llm_call(messages)
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
        args = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,   # Cap output — agent JSON fits in 2K; prevents 402 on low-credit accounts
            "response_format": {"type": "json_object"},
        }

        async def _do_call(current_args: dict) -> str:
            resp = await self.llm_client.chat.completions.create(**current_args)
            if not resp.choices:
                raise ValueError(f"No choices returned by {self.model}")
            content = resp.choices[0].message.content
            if not content or not content.strip():
                finish = getattr(resp.choices[0], "finish_reason", "unknown")
                raise ValueError(f"Empty content from {self.model} (finish_reason={finish})")
            return str(content)

        last_error: Exception | None = None

        # Attempt 1: vision + JSON mode
        try:
            return await _do_call(args)
        except Exception as e:
            last_error = e
            logger.warning(f"LLM attempt 1 ({self.model}): {e}")

        # Attempt 2: drop response_format (model may not support JSON mode)
        err_lower = str(last_error).lower()
        if any(kw in err_lower for kw in ("400", "response_format", "json_object", "json mode",
                                           "unsupported", "empty content", "no choices")):
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

    def _parse_agent_output(self, raw: str) -> AgentOutput | None:
        """Parse the LLM's JSON response into an AgentOutput model."""
        try:
            text = raw.strip()

            # 1. Try direct parsing
            try:
                data = json.loads(text)
                return AgentOutput(**data)
            except json.JSONDecodeError:
                pass

            # 2. Try extracting from markdown code block
            import re
            json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    return AgentOutput(**data)
                except json.JSONDecodeError:
                    pass

            # 3. Bracket-matching: find the outermost { ... } with proper nesting
            json_str = self._extract_outermost_json(text)
            if json_str:
                try:
                    data = json.loads(json_str)
                    return AgentOutput(**data)
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse LLM output. Raw content: {raw[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Error during agent output parsing: {e}")
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

            # Execute the action
            url_before = self.page.url
            result = await self._execute_action(action, browser_state)
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
                await self.page.goto(action.navigate.url)
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
                secs = action.wait.seconds
                if secs >= 8:
                    # Smart wait: watch for screen stability (AI done generating)
                    await self._wait_for_stable(max_seconds=secs)
                else:
                    await asyncio.sleep(secs)
                return ActionResult(action_name="wait", success=True)

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

    @staticmethod
    def _compress_screenshot(png_bytes: bytes) -> str:
        """
        Compress a screenshot from PNG to JPEG to reduce token usage.

        IMPORTANT: No resolution change! The LLM sees the image at the SAME
        resolution as the real screen, so coordinates match 1:1. This avoids
        the problem where the LLM estimates coordinates for a downscaled image
        but the mouse clicks at real-screen coordinates.

        JPEG at quality 80 is ~75% smaller than PNG for typical screenshots.
        """
        import base64
        try:
            from PIL import Image
            from io import BytesIO

            img = Image.open(BytesIO(png_bytes))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80, optimize=True)
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

    async def _wait_for_stable(
        self,
        max_seconds: int = 45,
        check_interval: float = 2.5,
        stable_needed: int = 2,
    ) -> None:
        """
        Wait until the screen stops changing — signals that an AI model (Grok,
        Claude, ChatGPT …) has finished generating its response.

        Algorithm:
          - Take a screenshot every `check_interval` seconds
          - Compare consecutive screenshots by MD5 hash
          - When the hash is identical for `stable_needed` consecutive checks → done
          - Abort after `max_seconds` regardless

        This replaces fixed-time waits and prevents the agent from interrupting
        an AI mid-response.
        """
        import hashlib

        prev_hash: str | None = None
        stable_count = 0
        start = time.time()
        elapsed = 0.0

        logger.info(f"⏳ Smart wait: watching for screen stability (up to {max_seconds}s)...")

        while elapsed < max_seconds:
            await asyncio.sleep(check_interval)
            elapsed = time.time() - start

            try:
                shot = await self.page.screenshot()
                curr_hash = hashlib.md5(shot).hexdigest()
            except Exception:
                break

            if curr_hash == prev_hash:
                stable_count += 1
                logger.info(f"   Screen stable ×{stable_count} ({elapsed:.0f}s elapsed)")
                if stable_count >= stable_needed:
                    logger.info("✅ Screen stable — response complete")
                    return
            else:
                if stable_count > 0:
                    logger.info(f"   Screen changed again — resetting stable counter")
                stable_count = 0

            prev_hash = curr_hash

        logger.info(f"⏳ Wait timeout ({max_seconds}s) — proceeding regardless")

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
            return ActionResult(
                action_name="computer_call",
                success=True,
                extracted_content=str(result) if result is not None else None,
            )
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
            raise ValueError(f"Unknown computer module '{module_name}'. "
                             f"Available: mouse, keyboard, clipboard, display")

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

    def _build_history_text(self) -> str:
        """Build a text summary of all previous steps for the agent_history section."""
        if not self.history:
            return ""

        lines = []
        for entry in self.history[-5:]:  # Last 5 steps to keep context manageable
            text = entry.to_history_text()
            # Include the agent's memory to help track conversation state
            if entry.agent_output.memory:
                text += f"\n  Memory: {entry.agent_output.memory}"
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
