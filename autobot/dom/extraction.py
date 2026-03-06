"""
DOM Extraction Service — Converts Playwright's accessibility tree into indexed elements.

Adapted from Browser Use's dom/service.py. Their version uses CDP directly via cdp_use.
We use Playwright's built-in accessibility snapshot, which gives us a similar tree
but through a simpler API.

The flow:
    1. page.accessibility.snapshot() → raw accessibility tree
    2. _build_element_tree() → converts to DOMElementNode tree with indices
    3. Result: DOMSerializedState with selector_map + LLM-ready text

Usage:
    service = DOMExtractionService(page)
    state = await service.extract()
    llm_text = state.llm_representation()  # Send this to the LLM
    element = state.selector_map[4]        # Get element by LLM's chosen index
"""
from __future__ import annotations

import logging
from typing import Any

from autobot.dom.models import (
    BrowserState,
    DOMElementNode,
    DOMSerializedState,
    PageInfo,
    SelectorMap,
    TabInfo,
)

logger = logging.getLogger(__name__)

# Tags that are always interactive (clickable/fillable)
INTERACTIVE_TAGS = {
    "a", "button", "input", "select", "textarea", "option",
    "details", "summary",
}

# Roles that indicate interactivity (from accessibility tree)
INTERACTIVE_ROLES = {
    "button", "link", "textbox", "combobox", "checkbox",
    "radio", "slider", "switch", "tab", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "spinbutton",
    "searchbox", "treeitem",
}

# Tags to skip entirely (no useful content for the LLM)
SKIP_TAGS = {
    "script", "style", "noscript", "br", "hr",
    "meta", "link", "head",
}


class DOMExtractionService:
    """
    Extracts the accessibility tree from a Playwright page and converts it
    into indexed interactive elements that the LLM can reference.

    Adapted from Browser Use's DOMService but using Playwright instead of CDP.
    """

    def __init__(self, page: Any, previous_state: DOMSerializedState | None = None):
        """
        Args:
            page: Playwright Page object.
            previous_state: Previous DOM state for detecting new elements.
        """
        self.page = page
        self.previous_state = previous_state
        self._index_counter = 1
        self._selector_map = SelectorMap()

    async def extract_state(self) -> BrowserState:
        """
        Extract complete browser state including DOM tree, page info, tabs, and screenshot.

        This is the main entry point — called every step of the agent loop.
        Returns a BrowserState that the prompt builder uses to create the LLM message.
        """
        # 1. Get accessibility tree
        try:
            ax_tree = await self.page.accessibility.snapshot()
        except Exception as e:
            logger.warning(f"Accessibility snapshot failed: {e}")
            ax_tree = None

        # 2. Build indexed element tree
        self._index_counter = 1
        self._selector_map = SelectorMap()

        element_tree = None
        if ax_tree:
            element_tree = self._build_element_tree(ax_tree, depth=0)

        # 3. Count element stats
        num_links = 0
        num_interactive = 0
        total_elements = 0
        if element_tree:
            stats = self._count_stats(element_tree)
            num_links = stats["links"]
            num_interactive = stats["interactive"]
            total_elements = stats["total"]

        # 4. Get page info (scroll position, viewport)
        page_info = await self._get_page_info()

        # 5. Get tab list
        tabs = await self._get_tabs()

        # 6. Take screenshot
        screenshot_b64 = None
        try:
            screenshot_bytes = await self.page.screenshot(type="png")
            import base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")

        # 7. Build serialized state
        serialized = DOMSerializedState(
            element_tree=element_tree,
            selector_map=self._selector_map,
        )

        return BrowserState(
            url=self.page.url,
            title=await self.page.title(),
            tabs=tabs,
            page_info=page_info,
            element_tree=element_tree,
            selector_map=self._selector_map,
            screenshot_b64=screenshot_b64,
            num_links=num_links,
            num_interactive=num_interactive,
            total_elements=total_elements,
        )

    def _build_element_tree(
        self,
        ax_node: dict[str, Any],
        depth: int = 0,
    ) -> DOMElementNode | None:
        """
        Recursively build a DOMElementNode tree from Playwright's accessibility snapshot.

        Playwright's accessibility.snapshot() returns a tree like:
        {
            "role": "WebArea",
            "name": "Google",
            "children": [
                {"role": "link", "name": "Gmail", ...},
                {"role": "textbox", "name": "Search", ...},
                {"role": "button", "name": "Google Search", ...},
            ]
        }

        We convert this into indexed DOMElementNodes that the LLM can reference.
        """
        role = ax_node.get("role", "")
        name = ax_node.get("name", "")
        value = ax_node.get("value", "")
        children_data = ax_node.get("children", [])

        # Map accessibility role to approximate HTML tag
        tag_name = _role_to_tag(role)

        if tag_name in SKIP_TAGS:
            return None

        # Determine if this element is interactive
        is_interactive = (
            role.lower() in INTERACTIVE_ROLES
            or tag_name in INTERACTIVE_TAGS
            or ax_node.get("focused", False)
        )

        # Build attributes dict from accessibility properties
        attributes: dict[str, str] = {}
        if role and role not in ("none", "generic", "WebArea", "text"):
            attributes["role"] = role
        if value:
            attributes["value"] = value
        if ax_node.get("checked") is not None:
            attributes["checked"] = str(ax_node["checked"])
        if ax_node.get("disabled"):
            attributes["disabled"] = "true"
        if ax_node.get("required"):
            attributes["required"] = "true"
        if ax_node.get("expanded") is not None:
            attributes["aria-expanded"] = str(ax_node["expanded"])
        if ax_node.get("level"):
            attributes["level"] = str(ax_node["level"])
        if ax_node.get("description"):
            attributes["aria-label"] = ax_node["description"]

        # Determine text content
        text = name or value or ""

        # Assign index for interactive elements
        index = None
        if is_interactive:
            index = self._index_counter
            self._index_counter += 1

        # Check if this element is new (didn't exist in previous state)
        is_new = False
        if index is not None and self.previous_state:
            # If same text+role wasn't in the previous selector map, it's new
            is_new = not self._was_in_previous_state(tag_name, text, role)

        # Create the node
        node = DOMElementNode(
            index=index,
            tag_name=tag_name,
            text=text,
            attributes=attributes,
            is_interactive=is_interactive,
            is_new=is_new,
            depth=depth,
            backend_id=f"ax_{self._index_counter}",
        )

        # Add to selector map if interactive
        if index is not None:
            self._selector_map[index] = node

        # Process children
        for child_data in children_data:
            child_node = self._build_element_tree(child_data, depth=depth + 1)
            if child_node:
                node.children.append(child_node)

        # Skip nodes that have no content and no interactive children
        if not is_interactive and not text.strip() and not node.children:
            return None

        return node

    def _was_in_previous_state(self, tag: str, text: str, role: str) -> bool:
        """Check if an element with similar properties existed in the previous DOM state."""
        if not self.previous_state or not self.previous_state.selector_map:
            return False

        for idx in range(1, len(self.previous_state.selector_map) + 1):
            prev_elem = self.previous_state.selector_map.get(idx)
            if prev_elem and prev_elem.tag_name == tag and prev_elem.text == text:
                return True
        return False

    def _count_stats(self, node: DOMElementNode) -> dict[str, int]:
        """Count element statistics for page_stats in the LLM prompt."""
        stats = {"links": 0, "interactive": 0, "total": 0}

        stats["total"] += 1
        if node.is_interactive:
            stats["interactive"] += 1
        if node.tag_name == "a":
            stats["links"] += 1

        for child in node.children:
            child_stats = self._count_stats(child)
            for key in stats:
                stats[key] += child_stats[key]

        return stats

    async def _get_page_info(self) -> PageInfo | None:
        """Get current scroll position and viewport dimensions."""
        try:
            metrics = await self.page.evaluate("""() => ({
                scrollY: window.scrollY,
                scrollX: window.scrollX,
                viewportHeight: window.innerHeight,
                viewportWidth: window.innerWidth,
                pageHeight: document.documentElement.scrollHeight,
                pageWidth: document.documentElement.scrollWidth,
            })""")

            return PageInfo(
                url=self.page.url,
                title=await self.page.title(),
                viewport_height=metrics["viewportHeight"],
                viewport_width=metrics["viewportWidth"],
                scroll_y=metrics["scrollY"],
                scroll_x=metrics["scrollX"],
                page_height=metrics["pageHeight"],
                page_width=metrics["pageWidth"],
            )
        except Exception as e:
            logger.warning(f"Page info extraction failed: {e}")
            return None

    async def _get_tabs(self) -> list[TabInfo]:
        """Get list of all open browser tabs."""
        tabs = []
        try:
            context = self.page.context
            for page in context.pages:
                tabs.append(TabInfo(
                    tab_id=str(hash(page))[-6:],
                    url=page.url,
                    title=await page.title() if not page.is_closed() else "closed",
                ))
        except Exception as e:
            logger.warning(f"Tab listing failed: {e}")
        return tabs


def _role_to_tag(role: str) -> str:
    """
    Map accessibility role to approximate HTML tag name.

    This is an approximation — the accessibility tree doesn't always
    give us the actual HTML tag, but we can infer it from the role.
    """
    role_map = {
        "WebArea": "html",
        "document": "html",
        "link": "a",
        "button": "button",
        "textbox": "input",
        "searchbox": "input",
        "combobox": "select",
        "checkbox": "input",
        "radio": "input",
        "slider": "input",
        "spinbutton": "input",
        "switch": "input",
        "tab": "button",
        "menuitem": "li",
        "menuitemcheckbox": "li",
        "menuitemradio": "li",
        "option": "option",
        "heading": "h2",
        "img": "img",
        "image": "img",
        "list": "ul",
        "listitem": "li",
        "table": "table",
        "row": "tr",
        "cell": "td",
        "paragraph": "p",
        "navigation": "nav",
        "banner": "header",
        "contentinfo": "footer",
        "complementary": "aside",
        "main": "main",
        "article": "article",
        "region": "section",
        "form": "form",
        "group": "div",
        "generic": "div",
        "none": "div",
        "separator": "hr",
        "dialog": "dialog",
        "alertdialog": "dialog",
        "alert": "div",
        "status": "div",
        "treeitem": "li",
        "tree": "ul",
        "grid": "table",
        "gridcell": "td",
        "rowheader": "th",
        "columnheader": "th",
        "text": "span",
        "StaticText": "span",
    }
    return role_map.get(role, "div")
