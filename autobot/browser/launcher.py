"""
Browser Launcher — Human-mode browser interaction.

Key design decisions:
- goto() navigates via Chrome's address bar (Ctrl+L) instead of webbrowser.open(),
  so it reuses the current tab instead of opening a new one.
- screenshot() focuses Chrome first via xdotool so the LLM always sees the browser,
  not whatever editor/terminal happens to be on top.
- All Playwright Page API methods are implemented so the agent loop doesn't crash.
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def _focus_chrome() -> None:
    """
    Bring the Chrome window to the foreground using xdotool (Linux).
    Falls back silently if xdotool is not installed.
    """
    try:
        # Try by class first (most reliable)
        r = subprocess.run(
            ["xdotool", "search", "--class", "Google-chrome",
             "windowactivate", "--sync"],
            timeout=3, capture_output=True,
        )
        if r.returncode != 0:
            # Fall back to window name
            subprocess.run(
                ["xdotool", "search", "--name", "Google Chrome",
                 "windowactivate", "--sync"],
                timeout=3, capture_output=True,
            )
    except FileNotFoundError:
        pass  # xdotool not installed — skip silently
    except Exception:
        pass


class _HumanKeyboard:
    """Keyboard emulation via PyAutoGUI, compatible with Playwright's keyboard API."""

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
        "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
        "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
        "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
        "Control": "ctrl", "Alt": "alt", "Shift": "shift", "Meta": "win",
    }

    async def press(self, key: str) -> None:
        """Press a key or key combo (e.g. 'Enter', 'Control+a', 'F5')."""
        import pyautogui
        if "+" in key:
            parts = [self._KEY_MAP.get(k.strip(), k.strip().lower()) for k in key.split("+")]
            pyautogui.hotkey(*parts)
        else:
            pyautogui_key = self._KEY_MAP.get(key, key.lower())
            pyautogui.press(pyautogui_key)
        await asyncio.sleep(0.1)

    async def type(self, text: str, delay: float = 30) -> None:
        """Type text using xdotool (handles all characters) with pyautogui fallback."""
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "30", text],
                timeout=max(10, len(text) // 10),
                capture_output=True,
            )
        except Exception:
            import pyautogui
            pyautogui.typewrite(text, interval=delay / 1000)
        await asyncio.sleep(0.1)


class _HumanContext:
    """Minimal Playwright BrowserContext emulation."""

    def __init__(self, page: "HumanModeEmulator") -> None:
        self._page = page

    @property
    def pages(self) -> list["HumanModeEmulator"]:
        return [self._page]

    async def new_page(self) -> "HumanModeEmulator":
        """Open a new browser tab via Ctrl+T (focuses Chrome first)."""
        import pyautogui
        _focus_chrome()
        await asyncio.sleep(0.3)
        pyautogui.hotkey("ctrl", "t")
        await asyncio.sleep(1.0)
        return self._page


class HumanModeEmulator:
    """
    Full Playwright Page-compatible interface backed by PyAutoGUI + xdotool.

    Key behaviours vs the previous version:
    - goto() uses Ctrl+L to navigate the CURRENT tab (no extra blank tab)
    - screenshot() focuses Chrome first so the LLM always sees the browser
    - All Playwright Page methods are implemented
    """

    def __init__(self) -> None:
        self.is_human_mode = True
        self._url: str = "about:blank"
        self._is_closed: bool = False
        self.keyboard = _HumanKeyboard()
        self._context_obj = _HumanContext(self)

    # ── Navigation ──────────────────────────────────────────────────────────

    async def goto(self, url: str, wait_until: str = "load", **kwargs) -> None:
        """
        Navigate to a URL by typing it into Chrome's address bar.

        Uses Ctrl+L to focus the address bar in the CURRENT tab, then types
        the URL and presses Enter. This avoids opening extra blank tabs that
        webbrowser.open() would create.
        """
        import pyautogui

        logger.info(f"→ Navigating to: {url}")

        # 1. Bring Chrome to front
        _focus_chrome()
        await asyncio.sleep(0.4)

        # 2. Focus the address bar
        pyautogui.hotkey("ctrl", "l")
        await asyncio.sleep(0.3)

        # 3. Clear existing content and type URL
        pyautogui.hotkey("ctrl", "a")
        await asyncio.sleep(0.1)

        # Use xdotool to type the URL reliably (handles ://?& etc.)
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", url],
                timeout=5, capture_output=True,
            )
        except Exception:
            pyautogui.typewrite(url, interval=0.03)

        await asyncio.sleep(0.2)
        pyautogui.press("enter")
        self._url = url

        # 4. Wait for page to load
        await asyncio.sleep(3.0)

    async def go_back(self, **kwargs) -> None:
        """Navigate back (Alt+Left)."""
        import pyautogui
        _focus_chrome()
        await asyncio.sleep(0.2)
        pyautogui.hotkey("alt", "left")
        await asyncio.sleep(1.0)

    async def go_forward(self, **kwargs) -> None:
        """Navigate forward (Alt+Right)."""
        import pyautogui
        _focus_chrome()
        await asyncio.sleep(0.2)
        pyautogui.hotkey("alt", "right")
        await asyncio.sleep(1.0)

    async def reload(self, **kwargs) -> None:
        """Reload page (F5)."""
        import pyautogui
        _focus_chrome()
        await asyncio.sleep(0.2)
        pyautogui.press("f5")
        await asyncio.sleep(2.5)

    # ── Observation ─────────────────────────────────────────────────────────

    async def screenshot(self, **kwargs) -> bytes:
        """
        Capture the screen as PNG bytes.

        Focuses Chrome first so the screenshot always shows the browser
        window, not whatever editor/terminal was last active.
        """
        import pyautogui
        from io import BytesIO

        _focus_chrome()
        await asyncio.sleep(0.5)   # Give Chrome time to paint after focus

        img = pyautogui.screenshot()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def evaluate(self, expression: str, **kwargs) -> Any:
        """
        Handle JS scroll expressions by routing to PyAutoGUI scroll.

        Supports: window.scrollBy(0, 300) → scroll down
                  window.scrollBy(0, -300) → scroll up
        """
        import pyautogui
        m = re.search(r"scrollBy\(\s*([^,]+),\s*([^)]+)\)", expression)
        if m:
            y_px = float(m.group(2).strip())
            pyautogui.scroll(-int(y_px / 100))
            await asyncio.sleep(0.3)
        return None

    async def wait_for_load_state(self, state: str = "load", **kwargs) -> None:
        await asyncio.sleep(2.0)

    async def wait_for_timeout(self, timeout: int = 1000) -> None:
        await asyncio.sleep(timeout / 1000)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def url(self) -> str:
        return self._url

    @property
    def context(self) -> _HumanContext:
        return self._context_obj

    @property
    def pages(self) -> list["HumanModeEmulator"]:
        return [self]

    def is_closed(self) -> bool:
        return self._is_closed

    async def bring_to_front(self) -> None:
        _focus_chrome()

    async def close(self) -> None:
        self._is_closed = True


class AsyncBrowserLauncher:
    """Provides a HumanModeEmulator as the agent's 'page' object."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    @classmethod
    def from_env(cls) -> "AsyncBrowserLauncher":
        return cls()

    async def start(self) -> HumanModeEmulator:
        logger.info("Initializing Human Mode (Vision-Only)...")
        return HumanModeEmulator()

    async def stop(self) -> None:
        pass
