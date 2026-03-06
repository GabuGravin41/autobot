"""
DOM Models — Data structures for representing browser DOM state.

Adapted from Browser Use's dom/views.py pattern:
- Simplified DOM nodes with element indices for LLM interaction
- Selector map: index → element for action execution
- Serialized DOM state for LLM representation

Key difference from Browser Use: We use Playwright's accessibility tree
instead of raw CDP, making this simpler and more portable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Attributes to include when serializing elements for the LLM.
# Adapted from Browser Use's DEFAULT_INCLUDE_ATTRIBUTES.
INCLUDE_ATTRIBUTES = [
    "title",
    "type",
    "name",
    "role",
    "value",
    "placeholder",
    "alt",
    "aria-label",
    "aria-expanded",
    "aria-checked",
    "checked",
    "selected",
    "disabled",
    "required",
    "href",
    "pattern",
    "min",
    "max",
    "minlength",
    "maxlength",
    "contenteditable",
]


@dataclass
class DOMElementNode:
    """
    A simplified DOM element for LLM consumption.

    Browser Use calls this SimplifiedNode. We keep it flat and simple:
    the LLM sees an index + tag + text + attributes, and clicks by index.
    """

    # Interactive index — this is what the LLM uses to reference the element.
    # Only set for interactive/clickable elements. None for pure text/structural elements.
    index: int | None

    # HTML tag name (e.g. "button", "input", "a", "div")
    tag_name: str

    # Visible text content of the element
    text: str

    # Relevant attributes (filtered to INCLUDE_ATTRIBUTES)
    attributes: dict[str, str]

    # Children elements
    children: list[DOMElementNode] = field(default_factory=list)

    # Is this element interactive (clickable, fillable, etc)?
    is_interactive: bool = False

    # Is this a new element that appeared since the last step?
    is_new: bool = False

    # Is this element scrollable?
    is_scrollable: bool = False

    # Depth in the tree (for indentation in LLM output)
    depth: int = 0

    # Playwright locator string for action execution
    selector: str = ""

    # Backend identifier for Playwright element handles
    backend_id: str = ""


@dataclass
class SelectorMap:
    """
    Maps element index → DOMElementNode.

    This is the critical bridge between the LLM's output ("click index 4")
    and the actual browser element. Browser Use calls this DOMSelectorMap.
    """

    _map: dict[int, DOMElementNode] = field(default_factory=dict)

    def __setitem__(self, index: int, element: DOMElementNode) -> None:
        self._map[index] = element

    def __getitem__(self, index: int) -> DOMElementNode:
        return self._map[index]

    def __contains__(self, index: int) -> bool:
        return index in self._map

    def get(self, index: int) -> DOMElementNode | None:
        return self._map.get(index)

    def __len__(self) -> int:
        return len(self._map)


@dataclass
class PageInfo:
    """Scroll position and viewport information for the current page."""
    url: str
    title: str
    viewport_height: int
    viewport_width: int
    scroll_y: int
    scroll_x: int
    page_height: int
    page_width: int

    @property
    def pixels_above(self) -> int:
        return self.scroll_y

    @property
    def pixels_below(self) -> int:
        return max(0, self.page_height - self.viewport_height - self.scroll_y)

    @property
    def pages_above(self) -> float:
        if self.viewport_height == 0:
            return 0
        return self.scroll_y / self.viewport_height

    @property
    def pages_below(self) -> float:
        if self.viewport_height == 0:
            return 0
        return self.pixels_below / self.viewport_height


@dataclass
class TabInfo:
    """Information about a browser tab."""
    tab_id: str
    url: str
    title: str


@dataclass
class BrowserState:
    """
    Complete browser state snapshot — sent to the LLM every step.

    Combines:
    - DOM tree as indexed interactive elements
    - Page/scroll info
    - Tab list
    - Screenshot (base64)
    """

    url: str
    title: str
    tabs: list[TabInfo]
    page_info: PageInfo | None
    element_tree: DOMElementNode | None
    selector_map: SelectorMap
    screenshot_b64: str | None = None

    # Stats for the LLM
    num_links: int = 0
    num_interactive: int = 0
    num_iframes: int = 0
    total_elements: int = 0


@dataclass
class DOMSerializedState:
    """
    The flattened text representation of the DOM, ready for the LLM prompt.

    Browser Use calls this SerializedDOMState.
    """

    # Root of the simplified element tree
    element_tree: DOMElementNode | None

    # Index → element mapping for action execution
    selector_map: SelectorMap

    def llm_representation(self, include_attributes: list[str] | None = None) -> str:
        """
        Convert the DOM tree into the compact text format the LLM reads.

        Output looks like:
            [1] <button> "Submit" (type=submit)
                [2] <input> "Search..." (placeholder=Search, type=text)
            [3] <a> "My Profile" (href=/profile)
            *[4] <div> "New popup" (role=dialog)   ← new since last step
        """
        if not self.element_tree:
            return "empty page"

        attrs_to_include = include_attributes or INCLUDE_ATTRIBUTES
        lines: list[str] = []
        self._serialize_node(self.element_tree, lines, attrs_to_include)
        return "\n".join(lines)

    def _serialize_node(
        self,
        node: DOMElementNode,
        lines: list[str],
        include_attributes: list[str],
    ) -> None:
        """Recursively serialize a DOM node and its children."""
        indent = "\t" * node.depth

        if node.is_interactive and node.index is not None:
            # Interactive element — gets an [index] marker
            new_marker = "*" if node.is_new else ""
            attr_str = self._format_attributes(node, include_attributes)
            text_preview = _cap_text(node.text, 80)
            text_part = f' "{text_preview}"' if text_preview else ""
            scroll_prefix = "|SCROLL| " if node.is_scrollable else ""

            lines.append(
                f"{indent}{scroll_prefix}{new_marker}[{node.index}] <{node.tag_name}>{text_part}{attr_str}"
            )
        elif node.text and node.text.strip():
            # Pure text node — no index
            text_preview = _cap_text(node.text.strip(), 120)
            if node.is_scrollable:
                lines.append(f"{indent}|SCROLL| {text_preview}")
            else:
                lines.append(f"{indent}{text_preview}")

        # Recurse into children
        for child in node.children:
            self._serialize_node(child, lines, include_attributes)

    @staticmethod
    def _format_attributes(node: DOMElementNode, include: list[str]) -> str:
        """Format element attributes for LLM display."""
        parts = []
        for attr_name in include:
            if attr_name in node.attributes and node.attributes[attr_name]:
                val = node.attributes[attr_name]
                # Truncate long attribute values
                if len(val) > 50:
                    val = val[:47] + "..."
                parts.append(f"{attr_name}={val}")
        if not parts:
            return ""
        return " (" + ", ".join(parts) + ")"


def _cap_text(text: str, max_len: int = 100) -> str:
    """Truncate text to max_len, appending '...' if truncated."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ").strip()
    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
