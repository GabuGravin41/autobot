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
        max_steps: int | None,
        agent_history: str | None = None,
        native_ui: str | None = None,
        page_snapshot=None,  # autobot.dom.page_snapshot.PageSnapshot | None
        memories: list[tuple[str, str]] | None = None,  # [(key, value), ...]
        click_zoom_b64: str | None = None,              # base64 JPEG crop around last click
        click_zoom_coords: tuple[int, int] | None = None,  # (x, y) of the click
        affordances: str | None = None,                 # per-step tool availability summary
        evolution_hint: str | None = None,              # dynamic trajectory correction from PromptEvolver
        remaining_hypotheses: list[str] | None = None,  # alternatives from previous failed step
    ):
        self.browser_state = browser_state
        self.task = task
        self.step_number = step_number
        self.max_steps = max_steps  # may be None in perpetual mode
        self.agent_history = agent_history
        self.native_ui = native_ui
        self.page_snapshot = page_snapshot
        self.memories = memories or []
        self.click_zoom_b64 = click_zoom_b64
        self.click_zoom_coords = click_zoom_coords
        self.affordances = affordances
        self.evolution_hint = evolution_hint
        self.remaining_hypotheses = remaining_hypotheses or []

    def _get_screen_size(self) -> tuple[int, int]:
        """Read screen resolution from the page title set by AgentLoop."""
        # AgentLoop encodes it as: "Human Mode | Screen: 1920×1080"
        import re
        m = re.search(r"Screen:\s*(\d+)×(\d+)", self.browser_state.title or "")
        if m:
            return int(m.group(1)), int(m.group(2))
        return 1920, 1080

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

        # 1. Recalled memories from previous runs
        if self.memories:
            mem_lines = "\n".join(f"  {k}: {v}" for k, v in self.memories)
            parts.append(
                f"<memory>\n"
                f"Facts remembered from previous sessions (use these to avoid repeating work):\n"
                f"{mem_lines}\n"
                f"</memory>"
            )

        # 2. Agent history
        if self.agent_history:
            parts.append(f"<agent_history>\n{self.agent_history}\n</agent_history>")

        # 2. Agent state
        date_str = datetime.now().strftime("%Y-%m-%d")
        agent_state = f"""<agent_state>
<user_request>
{self.task}
</user_request>
<step_info>Step {self.step_number + 1} of {self.max_steps if self.max_steps else '∞'} | Today: {date_str}</step_info>
</agent_state>"""
        parts.append(agent_state)

        # First-step situational awareness nudge (~30 tokens, step 1 only)
        if self.step_number == 0:
            parts.append(
                "<first_step_hint>This is your first observation. Assess what is on "
                "screen before acting — if the current page is not relevant to the "
                "task, navigate to your target.</first_step_hint>"
            )

        # 3. Browser state
        browser_state_text = self._build_browser_state()
        parts.append(f"<browser_state>\n{browser_state_text}\n</browser_state>")

        # 4. DOM snapshot — structured element list from Chrome DevTools (when available)
        if self.page_snapshot and self.page_snapshot.elements:
            snapshot_text = self.page_snapshot.to_prompt_text()
            parts.append(
                f"<dom_snapshot>\n"
                f"IMPORTANT: These are the REAL interactive elements on the current page, "
                f"extracted directly from the browser DOM. Use element numbers [N] to identify "
                f"what to click — coordinates from the screenshot are less reliable.\n\n"
                f"{snapshot_text}\n"
                f"</dom_snapshot>"
            )
        elif self.page_snapshot:
            # Snapshot succeeded but no interactive elements — still show page text
            parts.append(
                f"<dom_snapshot>\n"
                f"URL: {self.page_snapshot.url}\nTitle: {self.page_snapshot.title}\n"
                f"(No interactive elements detected — page may still be loading)\n"
                f"</dom_snapshot>"
            )

        # 5. Screen resolution (critical for coordinate estimation in Human Mode)
        screen_w, screen_h = self._get_screen_size()
        parts.append(
            f"<screen_info>\n"
            f"Resolution: {screen_w}×{screen_h} pixels\n"
            f"The screenshot is at FULL resolution — coordinates you see in the image match "
            f"the real screen pixels exactly. No scaling needed.\n"
            f"Chrome's content area is roughly: x=0–{screen_w}, y=80–{screen_h} "
            f"(top ~80px is the browser chrome/tab bar).\n"
            f"Center of screen: x={screen_w//2}, y={screen_h//2}\n"
            f"</screen_info>"
        )

        # 6. Per-step affordances — what tools are usable RIGHT NOW
        if self.affordances:
            parts.append(f"<affordances>\n{self.affordances}\n</affordances>")

        # 6b. Hypothesis options — alternatives from previous failed step still available
        if self.remaining_hypotheses:
            opts = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(self.remaining_hypotheses))
            parts.append(
                f"<hypothesis_options>\n"
                f"The previous step's chosen approach failed. These alternatives were NOT yet tried:\n"
                f"{opts}\n"
                f"Consider starting with option 1 unless you have a specific reason to choose differently.\n"
                f"</hypothesis_options>"
            )

        # 7. Dynamic trajectory correction (high-urgency hint from PromptEvolver)
        if self.evolution_hint:
            parts.append(self.evolution_hint)

        # 8. Native OS state
        if self.native_ui:
            parts.append(f"<native_os_state>\n{self.native_ui}\n</native_os_state>")

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

        # If a DOM snapshot (Chrome DevTools) is available, skip DOMSerializedState to
        # avoid sending the same element list twice (~600-1000 duplicate tokens per step).
        # The <dom_snapshot> block in build_text() is the single source of truth.
        if self.page_snapshot and self.page_snapshot.elements:
            return "\n".join(sections)

        # Fallback: no CDP snapshot — use Playwright element tree
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
            # Multi-part message with text + screenshot + optional click verification zoom
            content = [
                {"type": "text", "text": text_content},
                {"type": "text", "text": "Current screenshot:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.browser_state.screenshot_b64}",
                        "detail": "auto",
                    },
                },
            ]
            # Click zoom: a 300×300 crop around the last clicked point.
            # The LLM verifies whether the click landed on the correct target and can
            # immediately correct coordinates for the next click if needed.
            if self.click_zoom_b64 and self.click_zoom_coords:
                cx, cy = self.click_zoom_coords
                content.append({
                    "type": "text",
                    "text": (
                        f"CLICK VERIFICATION ZOOM — 300×300 region centred on last click "
                        f"at ({cx}, {cy}). Did it land on the right target? "
                        "If off, correct coordinates for next click."
                    ),
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.click_zoom_b64}",
                        "detail": "high",
                    },
                })
            return [{"role": "user", "content": content}]
        else:
            # Text-only message
            return [{"role": "user", "content": text_content}]
