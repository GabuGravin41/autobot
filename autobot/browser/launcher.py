"""
Async Browser — Connects to Chrome via CDP for DOM access.

Architecture (connect-first):
    1. Try to connect to an already-running Chrome on the CDP debug port
    2. If Chrome is running with CDP → attach via connect_over_cdp() → done
    3. If Chrome is NOT running at all → launch it with --remote-debugging-port
    4. NEVER try to launch a second Chrome when one is already running

Prerequisites:
    Chrome must be started with --remote-debugging-port=9222.
    The recommended way is to modify the Chrome desktop launcher:
        ~/.local/share/applications/google-chrome.desktop
    Change the Exec line to include --remote-debugging-port=9222

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
    Connects Playwright to an existing Chrome session via CDP.

    Primary mode: connect to already-running Chrome (user's real profile).
    Fallback mode: launch Chrome only if nothing is running.
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
        """Create launcher from environment variables."""
        return cls(
            debug_port=int(os.getenv("AUTOBOT_CDP_PORT", "9222")),
            chrome_path=os.getenv("AUTOBOT_CHROME_EXECUTABLE") or _detect_chrome(),
            user_data_dir=os.getenv("AUTOBOT_CHROME_USER_DATA_DIR") or _default_user_data_dir(),
            profile_dir=os.getenv("AUTOBOT_CHROME_PROFILE_DIR", "Default"),
        )

    async def start(self) -> Any:
        """
        Connect to Chrome via CDP and return a Playwright Page.

        Strategy:
            1. Check if CDP is available (Chrome running with --remote-debugging-port)
            2. If yes → connect directly (ideal case)
            3. If no, and Chrome is NOT running → launch Chrome with CDP
            4. If Chrome IS running but WITHOUT CDP → clear error with instructions

        Returns:
            Playwright Page object with full DOM access.
        """
        # Step 1: Try to connect to existing Chrome with CDP
        cdp_available = await self._is_cdp_available()
        if cdp_available:
            logger.info(f"CDP available on port {self.debug_port} — attaching to existing Chrome")
            await self._connect_playwright()
            logger.info(
                f"✅ Connected to Chrome via CDP (port {self.debug_port}). "
                f"Page: {self._page.url if self._page else 'none'}"
            )
            return self._page

        # Step 2: CDP not available — is Chrome running at all?
        chrome_running = self._is_chrome_running()

        if chrome_running:
            # Chrome is running but WITHOUT CDP — we cannot attach
            raise RuntimeError(
                f"\n"
                f"╔══════════════════════════════════════════════════════════════╗\n"
                f"║  Chrome is running but CDP (remote debugging) is NOT on.   ║\n"
                f"╚══════════════════════════════════════════════════════════════╝\n"
                f"\n"
                f"Autobot needs Chrome to be started with --remote-debugging-port={self.debug_port}\n"
                f"\n"
                f"Fix (one-time setup):\n"
                f"  1. Close ALL Chrome windows\n"
                f"  2. Run this in your terminal:\n"
                f"     google-chrome --remote-debugging-port={self.debug_port}\n"
                f"  3. Then retry your task in Autobot\n"
                f"\n"
                f"Or make it permanent by editing your Chrome launcher:\n"
                f"  cp /usr/share/applications/google-chrome.desktop ~/.local/share/applications/\n"
                f"  # Edit the Exec line to add --remote-debugging-port={self.debug_port}\n"
                f"\n"
                f"Verify CDP is working by visiting: http://localhost:{self.debug_port}/json\n"
            )

        # Step 3: Chrome is NOT running at all — launch it with CDP
        logger.info("No Chrome instance detected — launching Chrome with CDP enabled")
        await self._launch_chrome_fresh()
        await self._connect_playwright()
        logger.info(
            f"✅ Launched and connected to Chrome via CDP (port {self.debug_port}). "
            f"Page: {self._page.url if self._page else 'none'}"
        )
        return self._page

    def _is_chrome_running(self) -> bool:
        """Check if any Chrome process is currently running."""
        # Method 1: Check profile lock (most reliable)
        singleton_lock = Path(self.user_data_dir) / "SingletonLock"
        if singleton_lock.is_symlink() or singleton_lock.exists():
            return True

        lock_file = Path(self.user_data_dir) / "lockfile"
        if lock_file.exists():
            return True

        # Method 2: Check via pgrep (fallback)
        try:
            result = subprocess.run(
                ["pgrep", "-f", "chrome"],
                capture_output=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False

    async def _launch_chrome_fresh(self) -> None:
        """Launch a new Chrome process with --remote-debugging-port (only when no Chrome is running)."""
        if not self.chrome_path or not Path(self.chrome_path).exists():
            raise RuntimeError(
                f"Chrome not found at '{self.chrome_path}'. "
                "Set AUTOBOT_CHROME_EXECUTABLE in .env or install Chrome."
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
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to launch Chrome: {e}")

        # Wait for CDP to become available
        for attempt in range(25):
            if self._chrome_process:
                ret = self._chrome_process.poll()
                if ret is not None:
                    stderr_output = ""
                    try:
                        stderr_output = self._chrome_process.stderr.read().decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Chrome exited with code {ret}.\n"
                        f"{stderr_output}"
                    )

            if await self._is_cdp_available():
                logger.info(f"Chrome CDP ready after {attempt + 1}s")
                return

            if attempt > 5:
                logger.info(f"Waiting for CDP on port {self.debug_port} ({attempt + 1}/25)...")

            await _async_sleep(1.0)

        raise RuntimeError(
            f"Chrome launched but CDP not available on port {self.debug_port} after 25s.\n"
            "Try restarting your computer if this persists."
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

        # Use existing context and pages (user's real session)
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
            pages = self._context.pages
            if pages:
                # Use last page (most recent tab)
                self._page = pages[-1]
                logger.debug(f"Attached to existing page: {self._page.url}")
            else:
                self._page = await self._context.new_page()
                logger.debug("Created new page in existing context")
        else:
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            logger.debug("Created new context and page")

    async def _is_cdp_available(self, detailed: bool = False) -> bool | dict[str, Any]:
        """Check if CDP endpoint is responding on the debug port."""
        import httpx

        urls = [
            f"http://127.0.0.1:{self.debug_port}/json/version",
            f"http://localhost:{self.debug_port}/json/version",
        ]

        errors = []
        for url in urls:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
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
        """Allow agent loop to switch the active page."""
        self._page = new_page

    async def get_all_pages(self) -> list[Any]:
        """Get all open pages/tabs."""
        if self._context:
            return self._context.pages
        return []

    async def stop(self) -> None:
        """Disconnect Playwright (Chrome stays running)."""
        if self._browser:
            try:
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
        logger.info("Playwright disconnected from Chrome")

    async def ensure_page(self) -> Any:
        """Ensure we have a valid page, reconnecting if necessary."""
        if self._page and not self._page.is_closed():
            return self._page

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
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def _default_user_data_dir() -> str:
    """Default user data dir for Chrome profile."""
    # On Linux, use the real Chrome profile by default
    linux_path = Path.home() / ".config" / "google-chrome"
    if linux_path.exists():
        return str(linux_path)
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        return str(Path(local_app_data) / "Google" / "Chrome" / "User Data")
    return str(Path.home() / ".autobot" / "chrome_profile")


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)
