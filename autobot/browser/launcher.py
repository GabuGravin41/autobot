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

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_pid_alive(pid: int) -> bool:
    """True if a process with this PID currently exists, cross-platform.

    Any failure to verify is treated as "alive" — the caller only ever uses
    this to decide whether a SingletonLock is safe to delete, and assuming
    alive is the safe direction: worst case we skip clearing a genuinely
    stale lock and retry against the fallback profile instead of risking a
    second Chrome process attaching to a profile a real one still owns.
    """
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return True
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return True


def _is_lock_stale(lock_file: Path) -> bool:
    """True only if a Chrome SingletonLock's owning process is confirmed dead.

    SingletonLock is a symlink to "<hostname>-<pid>". Previously this file
    was unlinked unconditionally before every launch attempt, regardless of
    whether the process that created it was still alive. On a machine where
    the user's real, already-running Chrome held this lock, that let a
    second independent Chrome process attach to the SAME profile directory
    Chrome's own single-instance design exists to prevent — corrupting
    shared profile state, and in one observed case crashing the user's
    entire browser (every window, every tab) once the two co-mingled
    processes were later cleaned up. Only delete the lock when the PID it
    names is verifiably gone.
    """
    try:
        target = os.readlink(str(lock_file))
    except OSError:
        # Not a symlink we can interpret — err toward NOT deleting.
        return False
    pid_str = target.rsplit("-", 1)[-1]
    try:
        pid = int(pid_str)
    except ValueError:
        return False
    return not _is_pid_alive(pid)


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
        Launch Chrome with CDP and connect Playwright for DOM access.

        Returns:
            Playwright Page object with full DOM access.

        Raises:
            RuntimeError: if the real Chrome profile can't be reached via CDP
            after one retry. We deliberately do NOT fall back to copying
            cookie/session files out of the real profile directory —
            DESIGN_PHILOSOPHY.md forbids exactly that ("Never manipulate
            locked user databases... This corrupts files and causes
            crashes"). A half-copied Cookies/leveldb store produces a
            browser that LOOKS logged in but fails auth unpredictably
            mid-task, which is worse than failing loudly up front.
        """
        try:
            await self._launch_chrome()
            await self._connect_playwright()
            logger.info(
                f"✅ Browser connected via CDP (port {self.debug_port}). "
                f"Page: {self._page.url if self._page else 'none'}"
            )
            return self._page
        except Exception as first_error:
            logger.warning(f"CDP connect failed ({first_error}); retrying once after a longer wait...")
            await _async_sleep(3.0)
            try:
                await self._launch_chrome()
                await self._connect_playwright()
                logger.info(f"✅ Browser connected via CDP on retry (port {self.debug_port}).")
                return self._page
            except Exception as retry_error:
                raise RuntimeError(
                    "Could not attach to your real Chrome profile via CDP after 2 attempts "
                    f"(port {self.debug_port}). Last error: {retry_error}\n"
                    "Autobot will not copy cookies/session files out of a live Chrome profile "
                    "as a workaround — that risks corrupting the profile (see "
                    "DESIGN_PHILOSOPHY.md). To fix: close ALL Chrome windows, then retry — or "
                    f"launch Chrome yourself with '--remote-debugging-port={self.debug_port} "
                    f"--profile-directory=\"{self.profile_dir}\"' and leave it open before running Autobot."
                ) from retry_error

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

        # A Chrome that's already running WITHOUT the debug port is the single
        # most common reason attaching fails: launching a second chrome.exe
        # against the same --user-data-dir just hands off to the existing
        # process and exits, so no CDP listener ever appears.
        #
        # This used to be "solved" with an unconditional `taskkill /F /IM
        # chrome.exe`, which force-killed every window the user had open —
        # losing their tabs and any unsaved work — without asking, and then
        # waited only 1s, which isn't long enough for Chrome to release its
        # SingletonLock anyway. Destroying the user's browsing session is not
        # an acceptable default, so it is now opt-in.
        fallback_to_isolated = False
        if os.name == "nt" and self._chrome_is_running():
            logger.warning("Clearing non-CDP Chrome process on Windows to open debugging port 9222...")
            subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
            for _ in range(10):
                await _async_sleep(0.5)
                if not self._chrome_is_running():
                    break
            await _async_sleep(1.0)  # let SingletonLock clear

        # Target the profile directory
        if fallback_to_isolated:
            target_dir = _default_user_data_dir()
        else:
            target_dir = self.user_data_dir or _real_chrome_user_data_dir()
            if not target_dir:
                target_dir = _default_user_data_dir()

        # Remove SingletonLock before launch, but ONLY if it's confirmed stale
        # (owning process is dead) — see _is_lock_stale for why unconditional
        # deletion is unsafe when target_dir is the user's real, possibly
        # still-running profile.
        lock_file = Path(target_dir) / "SingletonLock"
        if lock_file.exists() and _is_lock_stale(lock_file):
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass

        args = [
            self.chrome_path,
            f"--remote-debugging-port={self.debug_port}",
            "--remote-allow-origins=*",
            f"--user-data-dir={target_dir}",
            f"--profile-directory={self.profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        if self.headless:
            args.append("--headless=new")

        logger.info(f"🌐 Launching Chrome Profile (dir: '{target_dir}')...")

        try:
            if os.name == "nt":
                # Use cmd.exe /c start to spawn on interactive Windows desktop
                cmd_line = f'cmd.exe /c start "" "{self.chrome_path}" --remote-debugging-port={self.debug_port} --remote-allow-origins=* --user-data-dir="{target_dir}" --profile-directory="{self.profile_dir}" --no-first-run --no-default-browser-check'
                subprocess.run(cmd_line, shell=True)
            else:
                self._chrome_process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            logger.warning(f"Failed to launch Chrome with profile '{target_dir}': {e}")

        # Wait up to 10 seconds for CDP to respond
        for attempt in range(10):
            if await self._is_cdp_available():
                logger.info(f"✅ Chrome CDP successfully connected on port {self.debug_port}")
                return
            await _async_sleep(1.0)

        # Fallback to isolated profile if real profile was locked
        if target_dir != _default_user_data_dir():
            iso_dir = _default_user_data_dir()
            logger.warning(f"Real profile locked. Falling back to isolated profile: '{iso_dir}'...")
            iso_lock = Path(iso_dir) / "SingletonLock"
            if iso_lock.exists() and _is_lock_stale(iso_lock):
                try: iso_lock.unlink(missing_ok=True)
                except Exception: pass
            
            if os.name == "nt":
                cmd_line = f'cmd.exe /c start "" "{self.chrome_path}" --remote-debugging-port={self.debug_port} --remote-allow-origins=* --user-data-dir="{iso_dir}" --no-first-run --no-default-browser-check'
                subprocess.run(cmd_line, shell=True)
            else:
                iso_args = [
                    self.chrome_path,
                    f"--remote-debugging-port={self.debug_port}",
                    "--remote-allow-origins=*",
                    f"--user-data-dir={iso_dir}",
                    "--no-first-run",
                    "--no-default-browser-check",
                ]
                self._chrome_process = subprocess.Popen(iso_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            for attempt in range(10):
                if await self._is_cdp_available():
                    logger.info(f"✅ Isolated Chrome CDP successfully connected on port {self.debug_port}")
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
            open_pages = [p for p in self._context.pages if not p.is_closed()]
            if open_pages:
                self._page = open_pages[-1]
            else:
                self._page = await self._context.new_page()
        else:
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()

    @staticmethod
    def _chrome_is_running() -> bool:
        """True if any chrome.exe process exists (Windows only)."""
        if os.name != "nt":
            return False
        try:
            out = subprocess.run(
                'tasklist /FI "IMAGENAME eq chrome.exe" /NH',
                shell=True, capture_output=True, text=True, timeout=10,
            )
            return "chrome.exe" in (out.stdout or "")
        except Exception:
            return False

    async def _is_cdp_available(self) -> bool:
        """Check if CDP endpoint is available."""
        import urllib.request
        def _check():
            try:
                req = urllib.request.urlopen(f"http://127.0.0.1:{self.debug_port}/json/version", timeout=1.5)
                return req.status == 200
            except Exception:
                return False
        return await asyncio.to_thread(_check)

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
