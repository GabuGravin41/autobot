"""
Window Control — OS-level window management and native UI interaction.

Uses Windows UI Automation (UIA) to interact with non-browser applications.
"""
from __future__ import annotations

import logging
import uiautomation as auto
from typing import List, Optional

from autobot.dom.native_extraction import NativeExtractionService

logger = logging.getLogger(__name__)

class Window:
    """Control native Windows applications and UI elements."""

    def __init__(self, mouse: Any, keyboard: Any):
        self._native_service = NativeExtractionService()
        self.mouse = mouse
        self.keyboard = keyboard

    def list_all(self) -> List[str]:
        """List all top-level window titles."""
        return [w.Name for w in auto.GetRootControl().GetChildren() if w.Name]

    def focus(self, title_query: str) -> bool:
        """Focus a window containing the given title text."""
        window = auto.WindowControl(searchDepth=1, Name=title_query)
        if window.Exists(0):
            window.SetFocus()
            window.SetActive()
            return True
        return False

    def extract_ui(self) -> str:
        """Extract the UI tree of the currently focused native window."""
        return self._native_service.extract_active_window()

    def click(self, index: int) -> bool:
        """Click a native UI element by its index from the last extraction."""
        control = self._native_service.get_element_by_index(index)
        if control:
            try:
                # Try UIA's specific pattern first (most robust)
                if hasattr(control, "Invoke"):
                    control.Invoke()
                else:
                    control.Click(simulateMove=True)
                return True
            except Exception as e:
                logger.warning(f"Native click on [{index}] failed: {e}")
                # Fallback to coordinate-based click if needed
                rect = control.BoundingRectangle
                if rect:
                    cx = rect.left + (rect.right - rect.left) // 2
                    cy = rect.top + (rect.bottom - rect.top) // 2
                    self.mouse.click(cx, cy)
                    return True
        return False

    def type(self, index: int, text: str) -> bool:
        """Type text into a native UI element by its index."""
        control = self._native_service.get_element_by_index(index)
        if control:
            try:
                if hasattr(control, "GetValuePattern"):
                    control.GetValuePattern().SetValue(text)
                else:
                    control.SetFocus()
                    self.keyboard.type(text)
                return True
            except Exception as e:
                logger.warning(f"Native type into [{index}] failed: {e}")
        return False
