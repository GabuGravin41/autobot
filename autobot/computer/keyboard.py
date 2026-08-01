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
        # Clamp timeout: 5s minimum, 30s maximum regardless of text length
        _timeout = max(5, min(30, len(text) // 20 + 5))
        try:
            # xdotool handles all characters reliably (URLs, Unicode, special chars)
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay",
                 str(int(interval * 1000)), text],
                timeout=_timeout,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            # Text too long for one shot — type in 200-char chunks
            logger.debug(f"xdotool type timed out, chunking {len(text)} chars")
            for _chunk in [text[i:i+200] for i in range(0, len(text), 200)]:
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay",
                     str(int(interval * 1000)), _chunk],
                    timeout=15,
                    capture_output=True,
                )
        except FileNotFoundError:
            # xdotool not installed — last resort pyautogui (ASCII only)
            logger.warning("xdotool not found — falling back to pyautogui (Unicode may not work)")
            import pyautogui
            pyautogui.typewrite(text, interval=interval)
        except Exception as e:
            logger.warning(f"xdotool type failed: {e}")
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

    # xdotool key names differ from pyautogui in some cases
    _XDOTOOL_KEY_MAP: dict[str, str] = {
        "enter": "Return", "return": "Return",
        "tab": "Tab",
        "esc": "Escape", "escape": "Escape",
        "backspace": "BackSpace",
        "delete": "Delete",
        "up": "Up", "down": "Down", "left": "Left", "right": "Right",
        "home": "Home", "end": "End",
        "pageup": "Prior", "pagedown": "Next",
        "space": "space",
        "ctrl": "ctrl", "alt": "alt", "shift": "shift",
        "f5": "F5", "f12": "F12",
    }

    def press(self, key: str) -> None:
        """Press a single key or key combo (e.g. 'Enter', 'ctrl+a', 'ctrl+shift+t').

        Uses xdotool for reliability — works even when the browser window is not
        the last-focused window, as xdotool sends events at the OS level.
        """
        import subprocess, time
        # Normalise to lowercase for mapping lookups
        if "+" in key:
            parts = [self._KEY_MAP.get(k.strip(), k.strip().lower()) for k in key.split("+")]
            xdotool_parts = [self._XDOTOOL_KEY_MAP.get(p, p) for p in parts]
            xdotool_key = "+".join(xdotool_parts)
        else:
            mapped = self._KEY_MAP.get(key, key.lower())
            xdotool_key = self._XDOTOOL_KEY_MAP.get(mapped, mapped)

        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", xdotool_key],
                timeout=5,
                capture_output=True,
            )
            logger.debug(f"Keyboard press (xdotool): {key} → {xdotool_key}")
            return
        except (FileNotFoundError, Exception) as e:
            logger.debug(f"xdotool key failed ({e}), falling back to pyautogui")

        import pyautogui
        if "+" in key:
            parts = [self._KEY_MAP.get(k.strip(), k.strip().lower()) for k in key.split("+")]
            pyautogui.hotkey(*parts)
        else:
            mapped = self._KEY_MAP.get(key, key.lower())
            pyautogui.press(mapped)
        logger.debug(f"Keyboard press (pyautogui): {key}")

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
