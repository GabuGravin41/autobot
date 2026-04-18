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
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
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

_CDP_PORT = 9222


def _cdp_port_open(host: str = "127.0.0.1", port: int = _CDP_PORT, timeout: float = 0.5) -> bool:
    """Return True if Chrome's remote-debugging port is actually reachable.

    This is the ONLY reliable check — the presence of a chrome process means
    nothing if the user launched Chrome without --remote-debugging-port.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _find_chrome_binary() -> str | None:
    """Return the path to a Chrome/Chromium binary, or None if not found."""
    explicit = os.getenv("AUTOBOT_CHROME_EXECUTABLE")
    if explicit and Path(explicit).exists():
        return explicit
    for binary in _CHROME_BINARIES:
        try:
            r = subprocess.run(["which", binary], capture_output=True, timeout=2)
            if r.returncode == 0:
                return r.stdout.decode().strip() or binary
        except Exception:
            pass
    return None


def _bootstrap_debug_profile(
    source_user_data_dir: Path,
    source_profile: str,
    debug_user_data_dir: Path,
    debug_profile: str,
) -> bool:
    """Create a standalone Chrome user-data-dir that copies one profile.

    Chrome 136+ refuses --remote-debugging-port on the real profile, so we
    build a dedicated dir containing only the profile we want to drive.
    Returns True if the debug profile is usable after this call.
    """
    debug_user_data_dir = debug_user_data_dir.expanduser()
    target_profile = debug_user_data_dir / debug_profile
    if target_profile.exists():
        return True  # already bootstrapped

    source_user_data_dir = source_user_data_dir.expanduser()
    src_profile = source_user_data_dir / source_profile
    if not src_profile.exists():
        logger.warning(
            f"Cannot bootstrap debug profile: source profile '{src_profile}' does not exist. "
            f"Chrome will launch with a fresh profile (you will need to log in)."
        )
        debug_user_data_dir.mkdir(parents=True, exist_ok=True)
        return False

    logger.info(
        f"Bootstrapping Chrome debug profile: copying '{src_profile}' → '{target_profile}' "
        f"(one-time copy, ~1-2 min depending on profile size)..."
    )
    debug_user_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy the profile folder itself
    try:
        shutil.copytree(src_profile, target_profile, symlinks=False, ignore_dangling_symlinks=True)
    except Exception as e:
        logger.error(f"Profile copy failed: {e}")
        return False

    # Copy the small top-level files Chrome expects (Local State, First Run)
    for fname in ("Local State", "First Run"):
        src_file = source_user_data_dir / fname
        if src_file.exists() and src_file.is_file():
            try:
                shutil.copy2(src_file, debug_user_data_dir / fname)
            except Exception as e:
                logger.debug(f"Skipped copying '{fname}': {e}")

    logger.info(f"✅ Debug profile ready at {debug_user_data_dir}")
    return True


def _launch_chrome() -> None:
    """Ensure a Chrome instance with remote-debugging is running.

    Behaviour:
      - If port 9222 already accepts connections → do nothing
      - Else, launch Chrome with --remote-debugging-port=9222 using a
        dedicated user-data-dir (Chrome 136+ forbids the default profile).
      - The profile to drive is taken from AUTOBOT_CHROME_SOURCE_PROFILE_DIR
        (copied once into AUTOBOT_CHROME_USER_DATA_DIR on first run).

    Relevant env vars (see .env):
      AUTOBOT_CHROME_USER_DATA_DIR        target debug dir (default: ~/.config/chrome-debug-profile)
      AUTOBOT_CHROME_PROFILE_DIR          profile name inside debug dir (default: Default)
      AUTOBOT_CHROME_SOURCE_USER_DATA_DIR source real Chrome dir (default: ~/.config/google-chrome)
      AUTOBOT_CHROME_SOURCE_PROFILE_DIR   source profile to clone (default: same as target)
      AUTOBOT_CHROME_EXECUTABLE           path to Chrome binary
      AUTOBOT_CHROME_LAUNCH_TIMEOUT_MS    how long to wait for CDP port (default: 15000)
    """
    if _cdp_port_open():
        logger.info(f"✅ Chrome CDP already live on port {_CDP_PORT} — reusing.")
        return

    # Detect any existing Chrome without debug port so we can warn the user.
    # We do NOT kill it — that would trash their session — but we must warn
    # because Chrome on Linux typically only allows one instance per profile.
    try:
        pg = subprocess.run(["pgrep", "-fi", "chrome"], capture_output=True, timeout=2)
        if pg.returncode == 0 and pg.stdout.strip():
            logger.warning(
                "Chrome is running but WITHOUT remote-debugging. Autobot will launch a "
                "separate Chrome instance using its own user-data-dir. Your original "
                "Chrome windows stay untouched."
            )
    except Exception:
        pass

    chrome_bin = _find_chrome_binary()
    if not chrome_bin:
        logger.error(
            "Chrome not found on PATH. Install google-chrome-stable or set "
            "AUTOBOT_CHROME_EXECUTABLE in .env."
        )
        return

    # Resolve profile config
    home = Path.home()
    debug_dir = Path(
        os.getenv("AUTOBOT_CHROME_USER_DATA_DIR") or (home / ".config/chrome-debug-profile")
    ).expanduser()
    debug_profile = os.getenv("AUTOBOT_CHROME_PROFILE_DIR") or "Default"
    src_dir = Path(
        os.getenv("AUTOBOT_CHROME_SOURCE_USER_DATA_DIR") or (home / ".config/google-chrome")
    ).expanduser()
    src_profile = os.getenv("AUTOBOT_CHROME_SOURCE_PROFILE_DIR") or debug_profile

    _bootstrap_debug_profile(src_dir, src_profile, debug_dir, debug_profile)

    cmd = [
        chrome_bin,
        f"--remote-debugging-port={_CDP_PORT}",
        f"--user-data-dir={debug_dir}",
        f"--profile-directory={debug_profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--new-window",
        "about:blank",
    ]
    logger.info(f"Launching Chrome: {' '.join(cmd)}")
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"Chrome launch failed: {e}")
        return

    # Wait for the CDP port to come up
    timeout_ms = int(os.getenv("AUTOBOT_CHROME_LAUNCH_TIMEOUT_MS", "15000"))
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if _cdp_port_open():
            logger.info(f"✅ Chrome CDP ready on port {_CDP_PORT} (profile='{debug_profile}')")
            return
        time.sleep(0.3)
    logger.warning(
        f"Chrome launched but CDP port {_CDP_PORT} did not open within {timeout_ms}ms. "
        "The agent will rely on screenshots only."
    )


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

        # Wait for the page to actually render. Modern SPAs (Grok, ChatGPT,
        # Overleaf) show a blank skeleton for 2-8s after the HTML finishes
        # loading. We check via CDP when available; otherwise we just wait
        # a fixed fallback period so the caller doesn't observe a blank page.
        _MIN_SPA_WAIT = 4.0   # always wait at least this long after navigation
        _MAX_SPA_WAIT = 8.0   # upper bound if CDP keeps saying "not ready"
        _CHECK_JS = (
            "JSON.stringify({"
            "  ready: document.readyState,"
            "  len: (document.body ? document.body.innerText.length : 0)"
            "})"
        )
        _cdp_ok = False
        _start = asyncio.get_event_loop().time()
        try:
            from autobot.dom.page_snapshot import _get_active_tab_ws_url, CDPClient
            for _i in range(int(_MAX_SPA_WAIT)):
                await asyncio.sleep(1.0)
                elapsed = asyncio.get_event_loop().time() - _start
                try:
                    ws = await asyncio.wait_for(_get_active_tab_ws_url(url_hint=url), timeout=1.0)
                    if not ws:
                        continue  # CDP not available yet — keep waiting, do not bail
                    _cdp_ok = True
                    cdp = CDPClient(ws)
                    await asyncio.wait_for(cdp.connect(), timeout=2.0)
                    try:
                        res = await asyncio.wait_for(
                            cdp.call("Runtime.evaluate", {"expression": _CHECK_JS, "returnByValue": True}),
                            timeout=3.0,
                        )
                    finally:
                        await cdp.close()
                    import json as _json
                    data = _json.loads(res.get("result", {}).get("value", "{}"))
                    if (
                        data.get("ready") == "complete"
                        and data.get("len", 0) > 200
                        and elapsed >= _MIN_SPA_WAIT
                    ):
                        logger.info(f"Page ready after {3 + _i + 1}s ({data['len']} chars): {url}")
                        break
                except Exception:
                    pass
        except Exception as _e:
            logger.debug(f"SPA readiness check setup failed (non-fatal): {_e}")
        if not _cdp_ok:
            logger.debug(f"CDP not reachable — used fixed {_MAX_SPA_WAIT}s wait after navigation to {url}")

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

    async def screenshot(self, focus: bool = True, **kwargs) -> bytes:
        """Capture the screen. Focuses Chrome by default so the agent actually sees it.

        focus=True (default): bring Chrome to front before capture. Use this for
            any browser-context observation. Without this, another window (IDE,
            terminal) may cover Chrome and the agent sees the wrong thing.
        focus=False: capture whatever is on screen right now. Use for monitoring
            desktop apps, terminals, or verifying window-switch results.
        """
        import pyautogui
        from io import BytesIO

        if focus:
            await asyncio.to_thread(_focus_chrome)
            await asyncio.sleep(0.35)  # window manager needs a moment to repaint
        else:
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
