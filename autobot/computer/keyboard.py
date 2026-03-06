"""
Keyboard Control — OS-level keyboard control via PyAutoGUI.

Adapted from Open Interpreter's computer/keyboard/keyboard.py.
Provides type, press, and hotkey methods.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class Keyboard:
    """Control the keyboard at the OS level."""

    def type(self, text: str, interval: float = 0.03) -> None:
        """Type text character by character.

        Args:
            text: The text to type.
            interval: Delay between each character in seconds.
        """
        import pyautogui
        pyautogui.typewrite(text, interval=interval)
        logger.debug(f"Keyboard type: '{text[:50]}...'")

    def write(self, text: str) -> None:
        """Type text using the write method (supports Unicode).

        Args:
            text: The text to type, supports Unicode characters.
        """
        import pyautogui
        pyautogui.write(text)
        logger.debug(f"Keyboard write: '{text[:50]}...'")

    def press(self, key: str) -> None:
        """Press a single key.

        Args:
            key: Key name, e.g., 'enter', 'tab', 'escape', 'space',
                 'backspace', 'delete', 'up', 'down', 'left', 'right',
                 'home', 'end', 'pageup', 'pagedown', 'f1'-'f12'.
        """
        import pyautogui
        pyautogui.press(key)
        logger.debug(f"Keyboard press: {key}")

    def hotkey(self, *keys: str) -> None:
        """Press a keyboard shortcut (multiple keys simultaneously).

        Args:
            *keys: Keys to press together, e.g., hotkey('ctrl', 'c')
                   for copy, hotkey('ctrl', 'shift', 't') for reopen tab.
        """
        import pyautogui
        pyautogui.hotkey(*keys)
        logger.debug(f"Keyboard hotkey: {'+'.join(keys)}")

    def key_down(self, key: str) -> None:
        """Hold a key down.

        Args:
            key: Key to hold down.
        """
        import pyautogui
        pyautogui.keyDown(key)
        logger.debug(f"Keyboard key_down: {key}")

    def key_up(self, key: str) -> None:
        """Release a held key.

        Args:
            key: Key to release.
        """
        import pyautogui
        pyautogui.keyUp(key)
        logger.debug(f"Keyboard key_up: {key}")
