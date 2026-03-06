"""
Clipboard Control — Read and write to the system clipboard.

Adapted from Open Interpreter's computer/clipboard/clipboard.py.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Clipboard:
    """Read and write the system clipboard."""

    def get(self) -> str:
        """Get the current clipboard contents.

        Returns:
            The text currently in the clipboard.
        """
        import pyautogui
        # PyAutoGUI doesn't have clipboard support directly;
        # we use pyperclip which pyautogui wraps on some platforms
        try:
            import pyperclip
            content = pyperclip.paste()
        except ImportError:
            # Fallback: use pyautogui's hotkey to copy and read
            import subprocess
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True,
            )
            content = result.stdout.strip()
        logger.debug(f"Clipboard get: '{content[:50]}...'")
        return content

    def set(self, text: str) -> None:
        """Set the clipboard contents.

        Args:
            text: The text to copy to the clipboard.
        """
        try:
            import pyperclip
            pyperclip.copy(text)
        except ImportError:
            import subprocess
            subprocess.run(
                ["powershell", "-command", f"Set-Clipboard -Value '{text}'"],
                capture_output=True,
            )
        logger.debug(f"Clipboard set: '{text[:50]}...'")

    def copy(self) -> str:
        """Press Ctrl+C and return the clipboard contents.

        Returns:
            The text that was copied.
        """
        import pyautogui
        pyautogui.hotkey("ctrl", "c")
        import time
        time.sleep(0.2)  # Wait for clipboard to update
        return self.get()

    def paste(self) -> None:
        """Press Ctrl+V to paste clipboard contents."""
        import pyautogui
        pyautogui.hotkey("ctrl", "v")
        logger.debug("Clipboard paste (Ctrl+V)")
