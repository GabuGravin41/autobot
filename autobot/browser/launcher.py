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
            user_data_dir=os.getenv("AUTOBOT_CHROME_USER_DATA_DIR") or _default_user_data_dir(),
            profile_dir=os.getenv("AUTOBOT_CHROME_PROFILE_DIR", "Default"),
        )

    async def start(self) -> Any:
        """
        Launch Chrome with CDP and connect Playwright.

        Returns:
            Playwright Page object with full DOM access.
        """
        # Step 0: Smart Connect - check if already running with CDP enabled
        cdp_res = await self._is_cdp_available(detailed=True)
        if isinstance(cdp_res, dict) and cdp_res.get("available"):
            logger.info(f"Smart Connect: Attaching to existing Chrome on port {self.debug_port}")
            await self._connect_playwright()
            return self._page

        # Step 1: Launch Chrome with debugging port
        await self._launch_chrome()

        # Step 2: Connect Playwright via CDP
        await self._connect_playwright()

        logger.info(
            f"✅ Browser connected via CDP (port {self.debug_port}). "
            f"Page: {self._page.url if self._page else 'none'}"
        )

        return self._page

    async def _launch_chrome(self) -> None:
        """Launch Chrome with --remote-debugging-port."""
        if not self.chrome_path or not Path(self.chrome_path).exists():
            raise RuntimeError(
                f"Chrome not found at '{self.chrome_path}'. "
                "Set AUTOBOT_CHROME_EXECUTABLE in .env or install Chrome."
            )

        # Fail fast: if profile is locked by a non-CDP Chrome process, we CANNOT hijack it.
        # This prevents the 25-second delay where Chrome "hangs" trying to delegate to the existing process.
        lock_file = Path(self.user_data_dir) / "lockfile"
        singleton_lock = Path(self.user_data_dir) / "SingletonLock"
        
        for lfile in [lock_file, singleton_lock]:
            if lfile.exists():
                try:
                    if os.name == "nt":
                        # On Windows, renaming a locked file raises PermissionError
                        lfile.rename(lfile)
                except PermissionError:
                    raise RuntimeError(
                        f"\n[!] CHROME PROFILE IN USE\n"
                        f"Your Chrome profile is currently OPEN. Autobot cannot connect unless you either:\n"
                        f"  1. Close all Chrome windows (including background apps in your system tray), OR\n"
                        f"  2. Relaunch your own Chrome WITH remote debugging enabled on port {self.debug_port}.\n"
                    )

        args = [
            self.chrome_path,
            f"--remote-debugging-port={self.debug_port}",
            f"--user-data-dir={self.user_data_dir}",
            f"--profile-directory={self.profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-allow-origins=*",
        ]

        if self.headless:
            args.append("--headless=new")

        logger.info(f"Launching Chrome: {' '.join(args[:3])}...")

        try:
            self._chrome_process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to launch Chrome: {e}")

        # Wait for Chrome to start and CDP to become available
        startup_msg_shown = False
        for attempt in range(25):
            # 1. Check if process is still alive
            if self._chrome_process:
                ret = self._chrome_process.poll()
                if ret is not None:
                    # Chrome closed immediately. Most common cause: Profile in use / Lock file
                    error_msg = f"Chrome process exited immediately with code {ret}."
                    
                    # Try to detect if it's a profile lock issue
                    lock_file = Path(self.user_data_dir) / "SingletonLock"
                    if lock_file.exists():
                        error_msg += (
                            f"\n[!] A profile lock was detected at: {lock_file}\n"
                            "This usually means another Chrome window is using this profile. "
                            "Please CLOSE ALL Chrome windows and try again, or use a dedicated profile."
                        )
                    raise RuntimeError(error_msg)

            # 2. Check CDP availability
            cdp_res = await self._is_cdp_available(detailed=True)
            if isinstance(cdp_res, dict) and cdp_res.get("available"):
                logger.info(f"Chrome CDP available after {attempt + 1} attempts")
                return
            
            cdp_info = cdp_res if isinstance(cdp_res, dict) else {"available": False}
            
            if not startup_msg_shown and attempt > 5:
                logger.info(f"Waiting for Chrome CDP on port {self.debug_port} (Attempt {attempt+1}/25)...")
                startup_msg_shown = True

            await _async_sleep(1.0)

        raise RuntimeError(
            f"Chrome launched (PID {self._chrome_process.pid if self._chrome_process else '?'}) "
            f"but CDP not available on port {self.debug_port} after 25s.\n"
            f"Diagnostic: {cdp_info.get('error', 'Unknown error')}\n"
            "Try restarting your computer if this persists, as the port might be ghosting."
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
                # Prefer the last page (most likely the one the user is looking at or was just opened)
                # But check if it's closed/empty first
                self._page = pages[-1]
                logger.debug(f"Connecting to existing page: {self._page.url}")
            else:
                self._page = await self._context.new_page()
                logger.debug("Created new page in existing context")
        else:
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            logger.debug("Created new context and page")

    async def _is_cdp_available(self, detailed: bool = False) -> bool | dict[str, Any]:
        """
        Check if CDP endpoint is available.
        
        Args:
            detailed: If True, returns a dict with diagnostic info.
        """
        import httpx
        # On some systems 127.0.0.1 works, on others localhost (IPv6) does
        urls = [
            f"http://127.0.0.1:{self.debug_port}/json/version",
            f"http://localhost:{self.debug_port}/json/version"
        ]

        errors = []
        for url in urls:
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return {"available": True} if detailed else True
                    errors.append(f"HTTP {resp.status_code} at {url}")
            except Exception as e:
                errors.append(f"{type(e).__name__} at {url}")
        
        return {"available": False, "error": " | ".join(errors)} if detailed else False

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
