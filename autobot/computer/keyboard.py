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
        """Type text character by character. Supports all characters including Unicode and special chars.

        Args:
            text: The text to type.
            interval: Delay between each character in seconds.
        """
        import subprocess
        try:
            # xdotool handles all characters reliably (URLs, Unicode, special chars)
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay",
                 str(int(interval * 1000)), text],
                timeout=max(10, len(text) // 5),
                capture_output=True,
            )
        except (FileNotFoundError, Exception) as e:
            # Fallback to pyautogui if xdotool not available
            logger.debug(f"xdotool type failed ({e}), falling back to pyautogui")
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

    # Map Playwright-style key names to pyautogui key names
    _KEY_MAP: dict[str, str] = {
        "Enter": "enter", "Return": "enter",
        "Tab": "tab",
        "Escape": "esc", "Esc": "esc",
        "Backspace": "backspace",
        "Delete": "delete",
        "ArrowUp": "up", "ArrowDown": "down",
        "ArrowLeft": "left", "ArrowRight": "right",
        "Home": "home", "End": "end",
        "PageUp": "pageup", "PageDown": "pagedown",
        "Space": "space", " ": "space",
        "Control": "ctrl", "Alt": "alt", "Shift": "shift", "Meta": "win",
    }

    def press(self, key: str) -> None:
        """Press a single key or key combo (e.g. 'enter', 'ctrl+a', 'ctrl+shift+t').

        Args:
            key: Key name or combo separated by '+'.
        """
        import pyautogui
        if "+" in key:
            # Combo key — split and use hotkey
            parts = [self._KEY_MAP.get(k.strip(), k.strip().lower()) for k in key.split("+")]
            pyautogui.hotkey(*parts)
        else:
            mapped = self._KEY_MAP.get(key, key.lower())
            pyautogui.press(mapped)
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
