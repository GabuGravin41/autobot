"""
Display Control — Screenshots and screen information.

Adapted from Open Interpreter's computer/display/display.py.
Provides screenshot capture and screen dimension queries.
"""
from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)


class Display:
    """Capture screenshots and get screen information."""

    def screenshot(self) -> str:
        """Take a screenshot of the entire screen, returns base64-encoded PNG.

        Returns:
            Base64-encoded PNG string of the current screen.
        """
        import pyautogui
        from io import BytesIO
        img = pyautogui.screenshot()
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug("Screenshot captured")
        return b64

    def screenshot_region(self, x: int, y: int, width: int, height: int) -> str:
        """Take a screenshot of a specific screen region, returns base64 PNG.

        Args:
            x: Left coordinate of the region.
            y: Top coordinate of the region.
            width: Width of the region.
            height: Height of the region.

        Returns:
            Base64-encoded PNG of the specified region.
        """
        import pyautogui
        from io import BytesIO
        img = pyautogui.screenshot(region=(x, y, width, height))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug(f"Screenshot region ({x},{y},{width},{height}) captured")
        return b64

    @staticmethod
    def size() -> tuple[int, int]:
        """Get the screen resolution.

        Returns:
            Tuple of (width, height) in pixels.
        """
        import pyautogui
        s = pyautogui.size()
        return (s.width, s.height)

    @staticmethod
    def width() -> int:
        """Get screen width in pixels."""
        import pyautogui
        return pyautogui.size().width

    @staticmethod
    def height() -> int:
        """Get screen height in pixels."""
        import pyautogui
        return pyautogui.size().height
