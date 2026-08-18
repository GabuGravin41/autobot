"""
DOM Extraction Service — Builds LLM-ready BrowserState from the live Chrome
DevTools Protocol snapshot (dom/page_snapshot.py), not from Playwright's
accessibility tree.

History: the original version of this file wrapped Playwright's
`page.accessibility.snapshot()`. It was deleted in commit 3b9c67b when the
project pivoted to CDP-based, coordinate-verified interaction (see
DESIGN_PHILOSOPHY.md's "CDP-first for DOM" rule and computer/browser.py's
click_element()/fill()), but the import in agent/loop.py was never updated,
so AgentLoop has been unable to import since.

This version fixes that AND closes a second gap: it builds its selector_map
from the exact same CDP query (dom/page_snapshot.py's _JS_EXTRACT) and the
exact same [N] indices that computer.browser.click_element()/fill() use to
act. Previously the plan (DOM extraction) and the act (element resolution)
were two independently-indexed systems that could silently drift apart —
one likely cause of "clicks the wrong thing." Now there is one snapshot,
one index space, one way to click.

Usage:
    service = DOMExtractionService(page)
    state = await service.extract_state()
    element = state.selector_map[4]        # Get element by LLM's chosen index
"""
from __future__ import annotations

import base64
import logging
from typing import Any

from autobot.dom.models import (
    BrowserState,
    DOMElementNode,
    DOMSerializedState,
    SelectorMap,
    TabInfo,
)
from autobot.dom.page_snapshot import get_page_snapshot

logger = logging.getLogger(__name__)

# Attributes copied onto DOMElementNode.attributes from the CDP structured
# element data, so DOMSerializedState._format_attributes() (dom/models.py)
# can render them for the LLM using the same INCLUDE_ATTRIBUTES it already knows.
_ATTR_FIELDS = ("role", "type", "name", "href")


class DOMExtractionService:
    """
    Extracts indexed interactive elements via CDP and packages them into a
    BrowserState for StepPromptBuilder — the Playwright `page` is used only
    for tab enumeration and screenshotting, never for DOM/element discovery
    or for resolving clicks (that stays in computer/browser.py, CDP-only).
    """

    def __init__(
        self,
        page: Any,
        previous_state: DOMSerializedState | None = None,
        capture_screenshot: bool = True,
    ):
        self.page = page
        self.previous_state = previous_state
        # When the caller already knows it will never send an image (vision
        # disabled), skip the capture entirely — it costs real wall-clock time
        # on every step for a result nothing reads.
        self.capture_screenshot = capture_screenshot

    async def extract_state(self) -> BrowserState:
        """
        Main entry point — called every step of the agent loop.
        Never raises: on CDP failure, returns an empty-but-valid BrowserState
        so the agent loop can still show the LLM "nothing is visible, retry"
        instead of crashing the whole run.
        """
        url_hint = self._safe_page_url()
        if self.page is None or (hasattr(self.page, "is_closed") and self.page.is_closed()):
            return self._empty_state(url_hint)

        try:
            snapshot = await get_page_snapshot(url_hint=url_hint)
        except Exception as e:
            logger.warning(f"CDP page snapshot raised: {e}")
            snapshot = None

        if snapshot is None:
            return self._empty_state(url_hint)

        prev_indices = (
            set(self.previous_state.selector_map._map.keys())
            if self.previous_state is not None
            else set()
        )

        selector_map = SelectorMap()
        children: list[DOMElementNode] = []
        num_links = 0
        num_inputs = 0

        for raw in snapshot.elements_data:
            node = self._build_node(raw)
            if node is None:
                continue
            node.is_new = bool(prev_indices) and node.index not in prev_indices
            selector_map[node.index] = node
            children.append(node)
            if node.tag_name == "a":
                num_links += 1
            if node.tag_name in ("input", "textarea") or node.attributes.get("contenteditable"):
                num_inputs += 1

        root = DOMElementNode(
            index=None,
            tag_name="body",
            text="",
            attributes={},
            children=children,
            is_interactive=False,
            depth=0,
        )

        tabs = await self._get_tabs()
        screenshot_b64 = await self._get_screenshot()

        return BrowserState(
            url=snapshot.url or url_hint or "",
            title=snapshot.title,
            tabs=tabs,
            page_info=None,  # CDP snapshot doesn't report scroll/viewport geometry
            element_tree=root,
            selector_map=selector_map,
            screenshot_b64=screenshot_b64,
            num_links=num_links,
            num_interactive=len(children),
            num_iframes=0,
            total_elements=len(children),
            page_text=snapshot.text or "",
        )

    def _build_node(self, raw: dict) -> DOMElementNode | None:
        """Convert one structured CDP element dict into a DOMElementNode."""
        index = raw.get("index")
        if index is None:
            return None

        attributes: dict[str, str] = {}
        for field in _ATTR_FIELDS:
            val = raw.get(field)
            if val:
                attributes[field] = str(val)
        if raw.get("disabled"):
            attributes["disabled"] = "true"
        if raw.get("is_content_editable"):
            attributes["contenteditable"] = "true"
        if raw.get("value"):
            attributes["value"] = str(raw["value"])

        return DOMElementNode(
            index=index,
            tag_name=raw.get("tag", "div"),
            text=raw.get("text", ""),
            attributes=attributes,
            children=[],
            is_interactive=True,
            depth=1,
        )

    def _safe_page_url(self) -> str | None:
        try:
            return self.page.url
        except Exception:
            return None

    async def _get_tabs(self) -> list[TabInfo]:
        try:
            pages = self.page.context.pages
        except Exception:
            return []

        tabs: list[TabInfo] = []
        for i, p in enumerate(pages):
            try:
                tabs.append(TabInfo(tab_id=str(hash(p))[-6:], url=p.url, title=await p.title()))
            except Exception:
                continue
        return tabs

    async def _get_screenshot(self) -> str | None:
        if not self.capture_screenshot:
            return None
        try:
            jpeg_bytes = await self.page.screenshot(type="jpeg", quality=60)
            return base64.b64encode(jpeg_bytes).decode("ascii")
        except Exception as e:
            logger.debug(f"Screenshot failed: {e}")
            return None

    def _empty_state(self, url_hint: str | None) -> BrowserState:
        return BrowserState(
            url=url_hint or "",
            title="",
            tabs=[],
            page_info=None,
            element_tree=DOMElementNode(
                index=None, tag_name="body", text="", attributes={}, children=[], depth=0
            ),
            selector_map=SelectorMap(),
            screenshot_b64=None,
            num_links=0,
            num_interactive=0,
            num_iframes=0,
            total_elements=0,
        )
