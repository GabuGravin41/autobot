"""
Mouse Control — OS-level mouse control via PyAutoGUI.

Adapted from Open Interpreter's computer/mouse/mouse.py.
Provides click, move, scroll, and position methods that the LLM
can use for OS-level automation (outside the browser).
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class Mouse:
    """Control the mouse cursor and clicks at the OS level."""

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        """Click at screen coordinates (x, y).

        Args:
            x: X coordinate on screen.
            y: Y coordinate on screen.
            button: 'left', 'right', or 'middle'.
            clicks: Number of clicks (2 for double-click).
        """
        import pyautogui
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        logger.debug(f"Mouse click at ({x}, {y}) button={button} clicks={clicks}")

    def double_click(self, x: int, y: int) -> None:
        """Double-click at screen coordinates."""
        self.click(x, y, clicks=2)

    def right_click(self, x: int, y: int) -> None:
        """Right-click at screen coordinates."""
        self.click(x, y, button="right")

    def move(self, x: int, y: int, duration: float = 0.3) -> None:
        """Move mouse to screen coordinates.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            duration: Time in seconds for the movement (0 = instant).
        """
        import pyautogui
        pyautogui.moveTo(x=x, y=y, duration=duration)
        logger.debug(f"Mouse move to ({x}, {y})")

    def scroll(self, clicks: int = 3) -> None:
        """Scroll the mouse wheel.

        Args:
            clicks: Positive = scroll up, Negative = scroll down.
        """
        import pyautogui
        pyautogui.scroll(clicks)
        logger.debug(f"Mouse scroll {clicks}")

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> None:
        """Drag from one position to another.

        Args:
            start_x: Starting X coordinate.
            start_y: Starting Y coordinate.
            end_x: Ending X coordinate.
            end_y: Ending Y coordinate.
            duration: Time for the drag operation.
        """
        import pyautogui
        pyautogui.moveTo(start_x, start_y)
        pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
        logger.debug(f"Mouse drag from ({start_x},{start_y}) to ({end_x},{end_y})")

    @staticmethod
    def position() -> tuple[int, int]:
        """Get current mouse cursor position.

        Returns:
            Tuple of (x, y) screen coordinates.
        """
        import pyautogui
        pos = pyautogui.position()
        return (pos.x, pos.y)
