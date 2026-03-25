"""
Browser Launcher — Human-mode browser interaction.

Key design decisions:
- goto() navigates via Chrome's address bar (Ctrl+L).
- screenshot() focuses Chrome first so the LLM always sees the browser.
- Tab management: tracks open tabs by index, switches via Ctrl+1-9 shortcuts.
- Chrome auto-launch: if Chrome isn't running, it's started with the user's
  default profile so all logins and cookies are available immediately.
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Chrome detection & auto-launch ──────────────────────────────────────────

_CHROME_BINARIES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
]


def _chrome_is_running() -> bool:
    """Return True if a Chrome/Chromium process is already running."""
    for name in ("chrome", "chromium", "Google Chrome"):
        try:
            r = subprocess.run(
                ["pgrep", "-fi", name],
                capture_output=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip():
                return True
        except Exception:
            pass
    return False


def _launch_chrome() -> None:
    """
    Launch Chrome with the user's default profile and remote-debugging port.

    Remote-debugging on 9222 is needed for CDP DOM snapshots.
    The user's real profile is used so all logins/cookies are available.
    If Chrome is already running, this is a no-op.
    """
    if _chrome_is_running():
        return

    logger.info("Chrome not detected — launching with user profile...")

    # Find available Chrome binary
    chrome_bin = None
    for binary in _CHROME_BINARIES:
        try:
            r = subprocess.run(["which", binary], capture_output=True, timeout=2)
            if r.returncode == 0:
                chrome_bin = binary
                break
        except Exception:
            pass

    if not chrome_bin:
        logger.warning(
            "Chrome not found. Please install Google Chrome and make sure it's on PATH. "
            "Continuing anyway — the agent will control whatever browser is on screen."
        )
        return

    try:
        subprocess.Popen([
            chrome_bin,
            "--new-window",
            "--remote-debugging-port=9222",
            "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.5)  # give Chrome time to open
        logger.info(f"✅ Chrome launched ({chrome_bin})")
    except Exception as e:
        logger.warning(f"Chrome launch failed: {e} — continuing anyway")


def _query_cdp_url() -> str | None:
    """Query Chrome CDP for the real URL of the frontmost tab.

    Returns None if CDP is unreachable (Chrome not running, or no debugging port).
    This is used by HumanModeEmulator.url to return the actual current URL even
    when the agent navigated via keyboard shortcuts instead of goto().
    """
    import json
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:9222/json", timeout=1) as resp:
            tabs = json.loads(resp.read())
        # Skip DevTools / extension / new-tab pages; return first real URL
        for tab in tabs:
            url = tab.get("url", "")
            if url and not url.startswith((
                "chrome-extension://", "devtools://", "chrome://newtab",
                "chrome://new-tab-page", "about:blank",
            )):
                return url
    except Exception:
        pass
    return None


def _focus_chrome() -> None:
    """Bring the Chrome window to the foreground using xdotool (Linux)."""
    try:
        r = subprocess.run(
            ["xdotool", "search", "--class", "Google-chrome",
             "windowactivate", "--sync"],
            timeout=3, capture_output=True,
        )
        if r.returncode != 0:
            subprocess.run(
                ["xdotool", "search", "--name", "Google Chrome",
                 "windowactivate", "--sync"],
                timeout=3, capture_output=True,
            )
    except FileNotFoundError:
        pass  # xdotool not installed
    except Exception:
        pass


# ── Keyboard emulation ────────────────────────────────────────────────────────

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


# ── Tab-aware browser context ─────────────────────────────────────────────────

class _HumanContext:
    """
    Minimal Playwright BrowserContext emulation with real tab tracking.

    Each call to new_page() opens Ctrl+T and creates a new HumanModeEmulator
    that tracks which tab index it occupies. switch_tab() uses Ctrl+1-9 to
    switch Chrome tabs reliably.
    """

    def __init__(self, initial_page: "HumanModeEmulator") -> None:
        self._pages: list["HumanModeEmulator"] = [initial_page]

    @property
    def pages(self) -> list["HumanModeEmulator"]:
        return list(self._pages)

    async def new_page(self) -> "HumanModeEmulator":
        """Open a new Chrome tab and return a HumanModeEmulator for it."""
        import pyautogui
        _focus_chrome()
        await asyncio.sleep(0.3)
        pyautogui.hotkey("ctrl", "t")
        await asyncio.sleep(1.2)  # wait for tab to open and paint

        new_tab_index = len(self._pages) + 1  # 1-based Chrome tab index
        new_page = HumanModeEmulator(
            tab_index=new_tab_index,
            context=self,
        )
        self._pages.append(new_page)
        logger.info(f"📑 New tab opened (tab {new_tab_index})")
        return new_page

    async def switch_to(self, page: "HumanModeEmulator") -> None:
        """Switch Chrome's active tab to the one matching the given page object."""
        import pyautogui
        tab_index = page.tab_index
        if 1 <= tab_index <= 8:
            _focus_chrome()
            await asyncio.sleep(0.2)
            pyautogui.hotkey("ctrl", str(tab_index))
            await asyncio.sleep(0.5)
            logger.info(f"📑 Switched to tab {tab_index}")
        else:
            # Tab 9+ — use Ctrl+9 (last tab) then navigate forward with Ctrl+Tab
            _focus_chrome()
            await asyncio.sleep(0.2)
            pyautogui.hotkey("ctrl", "9")
            extra_tabs = tab_index - 9
            for _ in range(extra_tabs):
                pyautogui.hotkey("ctrl", "tab")
                await asyncio.sleep(0.2)
            await asyncio.sleep(0.5)

    def remove(self, page: "HumanModeEmulator") -> None:
        if page in self._pages:
            self._pages.remove(page)


# ── Main emulator ─────────────────────────────────────────────────────────────

class HumanModeEmulator:
    """
    Full Playwright Page-compatible interface backed by PyAutoGUI + xdotool.

    Tracks its own tab_index so the context can switch to it by number.
    tab_index=1 means "the first/only tab" (default for the initial page).
    """

    def __init__(
        self,
        tab_index: int = 1,
        context: _HumanContext | None = None,
    ) -> None:
        self.is_human_mode = True
        self.tab_index = tab_index
        self._url: str = "about:blank"
        self._is_closed: bool = False
        self.keyboard = _HumanKeyboard()
        # Context is shared across all tabs from the same session
        self._context_obj: _HumanContext = context or _HumanContext(self)

    # ── Navigation ──────────────────────────────────────────────────────────

    async def goto(self, url: str, wait_until: str = "load", **kwargs) -> None:
        """Navigate to a URL by typing it into Chrome's address bar."""
        import pyautogui

        logger.info(f"→ Navigating to: {url}")

        # Ensure this tab is focused before navigating
        await self._context_obj.switch_to(self)

        # Focus the address bar
        pyautogui.hotkey("ctrl", "l")
        await asyncio.sleep(0.3)

        # Clear and type URL
        pyautogui.hotkey("ctrl", "a")
        await asyncio.sleep(0.1)
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
        await asyncio.sleep(3.0)

    async def go_back(self, **kwargs) -> None:
        await self._context_obj.switch_to(self)
        import pyautogui
        await asyncio.sleep(0.2)
        pyautogui.hotkey("alt", "left")
        await asyncio.sleep(1.0)

    async def go_forward(self, **kwargs) -> None:
        await self._context_obj.switch_to(self)
        import pyautogui
        await asyncio.sleep(0.2)
        pyautogui.hotkey("alt", "right")
        await asyncio.sleep(1.0)

    async def reload(self, **kwargs) -> None:
        await self._context_obj.switch_to(self)
        import pyautogui
        await asyncio.sleep(0.2)
        pyautogui.press("f5")
        await asyncio.sleep(2.5)

    # ── Observation ─────────────────────────────────────────────────────────

    async def screenshot(self, **kwargs) -> bytes:
        """Capture the full screen without changing focus.

        Intentionally does NOT call switch_to()/focus_chrome() — pyautogui
        captures whatever is currently on screen, which is exactly what the
        agent needs for desktop/terminal tasks and background monitoring.
        Chrome is already in focus when it matters because navigate/click/goto
        all call switch_to() before sending keyboard/mouse events.
        """
        import pyautogui
        from io import BytesIO

        await asyncio.sleep(0.2)
        img = pyautogui.screenshot()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def evaluate(self, expression: str, **kwargs) -> Any:
        """Handle JS scroll expressions by routing to PyAutoGUI scroll."""
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
        # Query Chrome's CDP for the real URL — catches keyboard navigation
        # (agent uses ctrl+l + type + Enter) that bypasses goto() and leaves
        # self._url stale as "about:blank".
        real = _query_cdp_url()
        if real:
            self._url = real   # keep cached value in sync
            return real
        return self._url

    @property
    def context(self) -> _HumanContext:
        return self._context_obj

    @property
    def pages(self) -> list["HumanModeEmulator"]:
        return self._context_obj.pages

    def is_closed(self) -> bool:
        return self._is_closed

    async def bring_to_front(self) -> None:
        await self._context_obj.switch_to(self)

    async def close(self) -> None:
        """Close this tab via Ctrl+W then remove from context."""
        import pyautogui
        await self._context_obj.switch_to(self)
        await asyncio.sleep(0.2)
        pyautogui.hotkey("ctrl", "w")
        await asyncio.sleep(0.5)
        self._is_closed = True
        self._context_obj.remove(self)

    # ── Stub methods expected by AgentLoop ──────────────────────────────────

    async def get_by_role(self, *args, **kwargs):
        raise NotImplementedError("Use computer.mouse.click() for human-mode interaction")

    async def get_by_text(self, *args, **kwargs):
        raise NotImplementedError("Use computer.mouse.click() for human-mode interaction")

    async def locator(self, *args, **kwargs):
        raise NotImplementedError("Use computer.mouse.click() for human-mode interaction")


# ── Launcher ──────────────────────────────────────────────────────────────────

class AsyncBrowserLauncher:
    """
    Provides a HumanModeEmulator as the agent's 'page' object.

    Auto-launches Chrome with the user's real profile if it isn't running.
    This preserves all existing logins, cookies, extensions, and saved passwords.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    @classmethod
    def from_env(cls) -> "AsyncBrowserLauncher":
        return cls()

    async def start(self) -> HumanModeEmulator:
        # Ensure Chrome is running before we try to control it
        await asyncio.to_thread(_launch_chrome)
        await asyncio.to_thread(_focus_chrome)
        logger.info("✅ Human Mode active — controlling your real Chrome profile.")
        return HumanModeEmulator(tab_index=1)

    async def stop(self) -> None:
        pass  # Chrome keeps running after the agent finishes (it's the user's browser)
