"""
Prompt Builder — Constructs the messages sent to the LLM each step.

Adapted from Browser Use's agent/prompts.py (AgentMessagePrompt class).
Builds structured messages with:
    - System message (from system_prompt.md template)
    - User message with <agent_state>, <browser_state>, <agent_history>
    - Optional screenshot as base64 image

This replaces the flat prompt in Autobot's original llm_brain.py._build_prompt().
"""
from __future__ import annotations

import importlib.resources
import logging
from datetime import datetime
from typing import Any

from autobot.dom.models import BrowserState, DOMSerializedState

logger = logging.getLogger(__name__)


class SystemPromptBuilder:
    """
    Loads and configures the system prompt from the markdown template.
    Adapted from Browser Use's SystemPrompt class.
    """

    def __init__(
        self,
        max_actions_per_step: int = 5,
        custom_instructions: str | None = None,
        tool_catalog: str = "",
    ):
        self.max_actions_per_step = max_actions_per_step
        self.custom_instructions = custom_instructions
        self.tool_catalog = tool_catalog
        self._template = self._load_template()

    def _load_template(self) -> str:
        """Load system prompt from the .md template file."""
        try:
            import os
            template_path = os.path.join(
                os.path.dirname(__file__), "system_prompt.md"
            )
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load system prompt template: {e}")
            return "You are a helpful browser automation agent."

    def build(self) -> str:
        """Build the final system prompt with template variables filled in."""
        prompt = self._template.format(
            max_actions=self.max_actions_per_step,
            tool_catalog=self.tool_catalog if self.tool_catalog else "",
        )
        if self.custom_instructions:
            prompt += f"\n\n# Additional Instructions\n{self.custom_instructions}"
        return prompt


class StepPromptBuilder:
    """
    Builds the user message for each step of the agent loop.

    Adapted from Browser Use's AgentMessagePrompt class.
    Each step message includes:
    - Agent history (what happened so far)
    - Agent state (task, step number)
    - Browser state (DOM tree, page stats, scroll position)
    - Screenshot (if available)
    """

    def __init__(
        self,
        browser_state: BrowserState,
        task: str,
        step_number: int,
        max_steps: int,
        agent_history: str | None = None,
    ):
        self.browser_state = browser_state
        self.task = task
        self.step_number = step_number
        self.max_steps = max_steps
        self.agent_history = agent_history

    def build_text(self) -> str:
        """
        Build the text portion of the user message.

        Output structure (following Browser Use's XML tag pattern):
            <agent_history>...</agent_history>
            <agent_state>
                <user_request>...</user_request>
                <step_info>...</step_info>
            </agent_state>
            <browser_state>
                <page_stats>...</page_stats>
                Tabs: ...
                Interactive elements:
                [1] <button> "Submit"
                ...
            </browser_state>
        """
        parts: list[str] = []

        # 1. Agent history
        if self.agent_history:
            parts.append(f"<agent_history>\n{self.agent_history}\n</agent_history>")

        # 2. Agent state
        date_str = datetime.now().strftime("%Y-%m-%d")
        agent_state = f"""<agent_state>
<user_request>
{self.task}
</user_request>
<step_info>Step {self.step_number + 1} of {self.max_steps} | Today: {date_str}</step_info>
</agent_state>"""
        parts.append(agent_state)

        # 3. Browser state
        browser_state_text = self._build_browser_state()
        parts.append(f"<browser_state>\n{browser_state_text}\n</browser_state>")

        return "\n\n".join(parts)

    def _build_browser_state(self) -> str:
        """
        Build the browser state description for the LLM.

        Adapted from Browser Use's AgentMessagePrompt._get_browser_state_description().
        """
        bs = self.browser_state
        sections: list[str] = []

        # Page stats
        stats = f"<page_stats>{bs.num_links} links, {bs.num_interactive} interactive, {bs.total_elements} total elements</page_stats>"
        if bs.total_elements < 10:
            stats = f"<page_stats>Page appears empty (SPA not loaded?) — {stats}</page_stats>"
        sections.append(stats)

        # Current tab
        sections.append(f"Current page: {bs.url}")
        sections.append(f"Title: {bs.title}")

        # Tab list
        if bs.tabs:
            tab_lines = []
            for tab in bs.tabs:
                title_preview = tab.title[:30] if tab.title else ""
                tab_lines.append(f"Tab {tab.tab_id}: {tab.url} - {title_preview}")
            sections.append("Available tabs:\n" + "\n".join(tab_lines))

        # Scroll / page info
        if bs.page_info:
            pi = bs.page_info
            pages_above = pi.pages_above
            pages_below = pi.pages_below
            sections.append(f"<page_info>{pages_above:.1f} pages above, {pages_below:.1f} pages below</page_info>")

        # Interactive elements (the DOM tree)
        serialized = DOMSerializedState(
            element_tree=bs.element_tree,
            selector_map=bs.selector_map,
        )
        elements_text = serialized.llm_representation()

        # Add start/end of page markers (from Browser Use)
        if bs.page_info:
            if bs.page_info.pages_above <= 0:
                elements_text = f"[Start of page]\n{elements_text}"
            if bs.page_info.pages_below <= 0:
                elements_text = f"{elements_text}\n[End of page]"

        if elements_text:
            sections.append(f"Interactive elements:\n{elements_text}")
        else:
            sections.append("Interactive elements:\nempty page")

        return "\n".join(sections)

    def build_messages(self, use_vision: bool = True) -> list[dict[str, Any]]:
        """
        Build the complete message list for the LLM API call.

        Returns a list of message dicts compatible with OpenAI's chat API:
        [{"role": "user", "content": [...]}]

        If use_vision=True and a screenshot is available, includes the image.
        """
        text_content = self.build_text()

        if use_vision and self.browser_state.screenshot_b64:
            # Multi-part message with text + image
            content = [
                {"type": "text", "text": text_content},
                {"type": "text", "text": "Current screenshot:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.browser_state.screenshot_b64}",
                        "detail": "auto",
                    },
                },
            ]
            return [{"role": "user", "content": content}]
        else:
            # Text-only message
            return [{"role": "user", "content": text_content}]
