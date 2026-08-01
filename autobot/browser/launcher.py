"""
Async Browser — Launches Chrome with CDP and connects Playwright for DOM access.

This is the critical bridge between the user's real Chrome profile and the new
agent loop. It solves the core problem: human_profile mode previously had NO DOM
access (BrowserController.start() was a no-op).

How it works:
    1. Launch Chrome with --remote-debugging-port=9222 using the user's real profile
    2. Connect Playwright via connect_over_cdp()
    3. Return a real Page object that works with dom/extraction.py

This is exactly what Browser Use does for their CDP connection.

Usage:
    launcher = AsyncBrowserLauncher()
    page = await launcher.start()
    # page is a Playwright Page with full DOM access + user's real cookies/sessions
    await launcher.stop()
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AsyncBrowserLauncher:
    """
    Launches Chrome with CDP debugging and connects Playwright for DOM access.

    Keeps the user's real profile (cookies, sessions, passwords) while giving
    Playwright full DOM access for the agent loop.
    """

    def __init__(
        self,
        debug_port: int = 9222,
        chrome_path: str | None = None,
        user_data_dir: str | None = None,
        profile_dir: str = "Default",
        headless: bool = False,
    ):
        self.debug_port = debug_port
        self.chrome_path = chrome_path or _detect_chrome()
        self.user_data_dir = user_data_dir or _default_user_data_dir()
        self.profile_dir = profile_dir
        self.headless = headless

        self._chrome_process: subprocess.Popen | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    @classmethod
    def from_env(cls) -> "AsyncBrowserLauncher":
        """Create launcher from environment variables (same vars as old BrowserController)."""
        return cls(
            debug_port=int(os.getenv("AUTOBOT_CDP_PORT", "9222")),
            chrome_path=os.getenv("AUTOBOT_CHROME_EXECUTABLE") or _detect_chrome(),
            user_data_dir=os.getenv("AUTOBOT_CHROME_USER_DATA_DIR") or _real_chrome_user_data_dir(),
            profile_dir=os.getenv("AUTOBOT_CHROME_PROFILE_DIR", "Default"),
        )

    async def start(self) -> Any:
        """
        Launch Chrome with CDP or native Playwright persistent context.

        Returns:
            Playwright Page object with full DOM access.
        """
        try:
            # Attempt 1: Try CDP launch & connection
            await self._launch_chrome()
            await self._connect_playwright()
            logger.info(
                f"✅ Browser connected via CDP (port {self.debug_port}). "
                f"Page: {self._page.url if self._page else 'none'}"
            )
            return self._page
        except Exception as e:
            logger.warning(f"CDP connection attempt hit limitation ({e}). Switching to Playwright Native Persistent Context...")
            return await self._launch_native_persistent_context()

    async def _launch_native_persistent_context(self) -> Any:
        """Launch Playwright Chromium using native persistent context with mirrored user session cookies."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        
        # Mirror real logged-in sessions (Cookies, Local Storage) into Autobot Automation profile
        real_dir = _real_chrome_user_data_dir()
        target_dir = _default_user_data_dir()
        if real_dir:
            self._mirror_user_session_stores(real_dir, target_dir)

        logger.info(f"🌐 Launching Playwright Persistent Context with Mirrored Sessions (dir: '{target_dir}')...")
        
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=target_dir,
            executable_path=self.chrome_path,
            headless=self.headless,
            args=["--no-first-run", "--no-default-browser-check"],
        )

        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()
        logger.info(f"✅ Mirrored Session Browser Connected! Page: {self._page.url}")
        return self._page

    def _mirror_user_session_stores(self, real_user_data_dir: str, target_user_data_dir: str) -> None:
        """Merge active session stores across ALL user Chrome profiles into Autobot Automation profile."""
        import shutil

        real_base = Path(real_user_data_dir)
        target_default = Path(target_user_data_dir) / "Default"

        if not real_base.exists():
            return

        target_default.mkdir(parents=True, exist_ok=True)

        session_files = [
            "Cookies",
            "Cookies-journal",
            "Network/Cookies",
            "Network/Cookies-journal",
            "Local Storage",
            "Session Storage",
            "IndexedDB",
        ]

        # Scan all profile subfolders (Default, Profile 1, Profile 2, etc.)
        profile_folders = [f for f in real_base.iterdir() if f.is_dir() and (f.name == "Default" or f.name.startswith("Profile "))]

        copied = 0
        for prof_folder in profile_folders:
            for rel_p in session_files:
                src = prof_folder / rel_p
                dst = target_default / rel_p
                if src.exists():
                    try:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if src.is_dir():
                            # Merge directories without overwriting if already populated by a richer profile
                            for sub in src.rglob("*"):
                                if sub.is_file():
                                    sub_rel = sub.relative_to(src)
                                    target_file = dst / sub_rel
                                    if not target_file.exists():
                                        target_file.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy2(sub, target_file)
                        else:
                            if not dst.exists():
                                shutil.copy2(src, dst)
                        copied += 1
                    except Exception as e:
                        logger.debug(f"Could not mirror {rel_p} from {prof_folder.name}: {e}")

        logger.info(f"🔄 Merrored {copied} session stores across {len(profile_folders)} Chrome profiles into Autobot master session!")

    async def _launch_chrome(self) -> None:
        """Launch Chrome with --remote-debugging-port."""
        if not self.chrome_path or not Path(self.chrome_path).exists():
            raise RuntimeError(
                f"Chrome not found at '{self.chrome_path}'. "
                "Set AUTOBOT_CHROME_EXECUTABLE in .env or install Chrome."
            )

        # Check if Chrome is already running with CDP on this port
        if await self._is_cdp_available():
            logger.info(f"Chrome already running with CDP on port {self.debug_port}")
            return

        # Attempt to kill lingering non-CDP Chrome processes on Windows to release profile lock
        try:
            if os.name == "nt":
                subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
                logger.info("🧹 Closed background Chrome processes to release profile lock for CDP.")
                await _async_sleep(1.0)
        except Exception as e:
            logger.debug(f"Taskkill check: {e}")

        # Target the real user profile directory strictly
        target_dir = self.user_data_dir or _real_chrome_user_data_dir()
        if not target_dir:
            target_dir = _default_user_data_dir()

        args = [
            self.chrome_path,
            f"--remote-debugging-port={self.debug_port}",
            f"--user-data-dir={target_dir}",
            f"--profile-directory={self.profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        if self.headless:
            args.append("--headless=new")

        logger.info(f"🌐 Launching Real Chrome Profile (dir: '{target_dir}')...")

        try:
            self._chrome_process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to launch Chrome with profile '{target_dir}': {e}")

        # Wait up to 15 seconds for CDP to respond
        for attempt in range(15):
            if await self._is_cdp_available():
                logger.info(f"✅ Real Chrome CDP successfully connected on port {self.debug_port}")
                return
            await _async_sleep(1.0)

        raise RuntimeError(
            f"Could not connect to Chrome CDP on port {self.debug_port}. "
            "Please close open Chrome instances or run Chrome with --remote-debugging-port=9222."
        )

    async def _connect_playwright(self) -> None:
        """Connect Playwright to Chrome via CDP."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{self.debug_port}",
                timeout=10000,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect Playwright via CDP on port {self.debug_port}: {e}"
            )

        # Get existing context and page, or create new ones
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
            pages = self._context.pages
            if pages:
                self._page = pages[0]
            else:
                self._page = await self._context.new_page()
        else:
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()

    async def _is_cdp_available(self) -> bool:
        """Check if CDP endpoint is available."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"http://127.0.0.1:{self.debug_port}/json/version")
                return resp.status_code == 200
        except Exception:
            return False

    @property
    def page(self) -> Any:
        """Current active page."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    @page.setter
    def page(self, new_page: Any) -> None:
        """Allow agent loop to switch the active page (e.g. after new_tab)."""
        self._page = new_page

    async def get_all_pages(self) -> list[Any]:
        """Get all open pages/tabs."""
        if self._context:
            return self._context.pages
        return []

    async def stop(self) -> None:
        """Disconnect Playwright and optionally close Chrome."""
        if self._browser:
            try:
                # Disconnect without closing the browser (user's Chrome stays open)
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

        self._page = None
        self._context = None

        # Note: We do NOT kill the Chrome process — user's Chrome stays running
        logger.info("Playwright disconnected from Chrome")

    async def ensure_page(self) -> Any:
        """Ensure we have a valid page, reconnecting if necessary."""
        if self._page and not self._page.is_closed():
            return self._page

        # Page was closed, try to get another one
        if self._context:
            pages = self._context.pages
            if pages:
                self._page = pages[-1]
                return self._page

        # Reconnect entirely
        await self._connect_playwright()
        return self._page


def _detect_chrome() -> str | None:
    """Detect Chrome executable path."""
    paths = [
        os.getenv("CHROME_EXECUTABLE"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
    ]
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def _default_user_data_dir() -> str:
    """Default user data dir for Autobot's Chrome profile."""
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        return str(Path(local_app_data) / "Autobot" / "ChromeAutomationProfile")
    return str(Path.home() / ".autobot" / "chrome_profile")


def _real_chrome_user_data_dir() -> str | None:
    """Get the user's real Chrome user data directory."""
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        path = Path(local_app_data) / "Google" / "Chrome" / "User Data"
        if path.exists():
            return str(path)
    home = Path.home()
    mac_path = home / "Library" / "Application Support" / "Google" / "Chrome"
    if mac_path.exists():
        return str(mac_path)
    linux_path = home / ".config" / "google-chrome"
    if linux_path.exists():
        return str(linux_path)
    return None


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)
