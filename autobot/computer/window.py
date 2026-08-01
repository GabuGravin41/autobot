"""
Window Control — OS-level window management and native UI interaction.

Uses Windows UI Automation (UIA) to interact with non-browser applications.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import uiautomation as auto  # Windows-only; callers must guard this import

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

    def active_title(self) -> str:
        """Return the title of the currently focused window.

        Used by the agent loop to decide whether it is looking at the browser
        (in which case the DOM snapshot already describes the screen) or at a
        native app (in which case it needs extract_ui()).
        """
        try:
            window = auto.GetForegroundWindow()
            if isinstance(window, int):
                window = auto.ControlFromHandle(window)
            return getattr(window, "Name", "") or ""
        except Exception as e:
            logger.debug(f"active_title failed: {e}")
            return ""

    def extract_ui(self) -> str:
        """Extract the UI tree of the currently focused native window.

        Element indices in the returned tree are the ones click()/type() below
        resolve against — they share this instance's selector map, so always
        extract and act through the same Window object rather than creating a
        separate NativeExtractionService.
        """
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
