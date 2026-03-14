"""
Native Extraction Service — Converts Windows UI Automation tree into indexed elements.

This is the OS-level equivalent of DOMExtractionService. It uses the 
Microsoft UI Automation (UIA) API to traverse the window tree of 
native applications and extract interactive elements (buttons, inputs, etc.).

Usage:
    service = NativeExtractionService()
    state_text = service.extract_active_window()
    element = service.get_element_by_index(5)
"""
from __future__ import annotations

import logging
import platform
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# uiautomation is Windows-only (COM/UIA). Guard the import so Linux doesn't crash.
_IS_WINDOWS = platform.system() == "Windows"
auto = None  # type: ignore
INTERACTIVE_CONTROL_TYPES: set = set()

if _IS_WINDOWS:
    try:
        import uiautomation as auto  # type: ignore
        INTERACTIVE_CONTROL_TYPES = {
            auto.ControlType.ButtonControl,
            auto.ControlType.EditControl,
            auto.ControlType.ListItemControl,
            auto.ControlType.MenuItemControl,
            auto.ControlType.CheckBoxControl,
            auto.ControlType.RadioButtonControl,
            auto.ControlType.ComboBoxControl,
            auto.ControlType.HyperlinkControl,
            auto.ControlType.TabItemControl,
            auto.ControlType.TreeItemControl,
        }
    except ImportError:
        logger.warning("uiautomation not available — native extraction disabled")
else:
    logger.debug("Native extraction (UIA) is Windows-only — skipped on %s", platform.system())

class NativeElementNode:
    def __init__(
        self,
        index: Optional[int],
        control: auto.Control,
        control_type: str,
        name: str,
        value: str,
        depth: int,
    ):
        self.index = index
        self.control = control
        self.control_type = control_type
        self.name = name
        self.value = value
        self.depth = depth
        self.children: List[NativeElementNode] = []

    def llm_representation(self) -> str:
        """Returns a string representing this element for the LLM."""
        prefix = "  " * self.depth
        idx_str = f"[{self.index}] " if self.index is not None else ""
        content = f"{self.name}"
        if self.value and self.value != self.name:
            content += f" (value: {self.value})"
        
        return f"{prefix}{idx_str}<{self.control_type}> {content}"

class NativeExtractionService:
    """
    Extracts the UI tree from the active Windows application using UIA.
    """

    def __init__(self):
        self._index_counter = 1
        self._selector_map: Dict[int, Any] = {}
        if _IS_WINDOWS and auto is not None:
            auto.uiautomation.SetGlobalSearchTimeout(1.0)

    def extract_active_window(self) -> str:
        """
        Extract the UI tree of the currently focused window.
        Returns a formatted string for the LLM.
        """
        self._index_counter = 1
        self._selector_map = {}

        if not _IS_WINDOWS or auto is None:
            return "Native extraction is only available on Windows."

        try:
            window = auto.GetForegroundWindow()
            if not window:
                return "No active window found."
            
            # If GetForegroundWindow returns a handle (int), wrap it
            if isinstance(window, int):
                window = auto.ControlFromHandle(window)

            root_node = self._build_node(window, depth=0)
            if not root_node:
                return "Active window has no readable elements."

            return self._serialize_tree(root_node)
        except Exception as e:
            logger.error(f"Native extraction failed: {e}")
            return f"Error extracting native UI: {e}"

    def get_element_by_index(self, index: int) -> Optional[auto.Control]:
        """Retrieve a UIA control by its LLM-assigned index."""
        return self._selector_map.get(index)

    def _build_node(self, control: Any, depth: int) -> Optional[NativeElementNode]:
        """Recursively build a tree of NativeElementNodes."""
        # Ensure we have a valid Control object
        if not hasattr(control, "ControlTypeName") or not hasattr(control, "ControlType"):
            return None

        # Limit depth to prevent infinite loops or massive trees
        if depth > 10:
            return None

        control_type = control.ControlTypeName.replace("Control", "").lower()
        name = control.Name or ""
        
        # Get value if applicable (e.g. for Edit controls)
        value = ""
        try:
            if hasattr(control, "GetValuePattern"):
                value = control.GetValuePattern().Value or ""
        except:
            pass

        # Determine if interactive
        is_interactive = control.ControlType in INTERACTIVE_CONTROL_TYPES
        
        index = None
        if is_interactive and name:
            index = self._index_counter
            self._selector_map[index] = control
            self._index_counter += 1

        node = NativeElementNode(
            index=index,
            control=control,
            control_type=control_type,
            name=name,
            value=value,
            depth=depth,
        )

        # Optimization: Only recurse into containers or if we haven't found many elements yet
        # Native trees can be extremely deep and noisy.
        try:
            for child in control.GetChildren():
                child_node = self._build_node(child, depth + 1)
                if child_node:
                    node.children.append(child_node)
        except:
            pass

        # Prune branches that have no text and no interactive children
        if not is_interactive and not name.strip() and not node.children:
            return None

        return node

    def _serialize_tree(self, node: NativeElementNode) -> str:
        """Converts the node tree into a flat string."""
        lines = [node.llm_representation()]
        for child in node.children:
            lines.append(self._serialize_tree(child))
        return "\n".join(lines)
