"""
Browser Launcher — Handles human-mode browser interaction.

The user has explicitly requested the removal of CDP (Chrome DevTools Protocol)
due to persistent port conflicts and complexity on Linux.

Architecture:
- Human Mode: Uses vision-only observation (screenshots) and OS-level control.
- Interface: Mimics a Playwright Page for compatibility with the agent loop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class HumanModeEmulator:
    """
    Standard browser interface for Autobot.
    
    Instead of CDP/Playwright, it uses:
    - webbrowser to open URLs in the user's real Chrome profile.
    - pyautogui for screenshots (vision).
    - OS-level coordinates for interaction (controlled by AgentLoop).
    """
    def __init__(self):
        self.is_human_mode = True
        self._url = "about:blank"
        self._is_closed = False

    async def goto(self, url: str, **kwargs) -> None:
        """Open a URL in the user's default browser."""
        import webbrowser
        logger.info(f"Human Mode: Opening {url}")
        # webbrowser.open is non-blocking but opens the URL in the background
        webbrowser.open(url)
        self._url = url
        # Wait for the browser to open and render
        await asyncio.sleep(2.0)

    async def screenshot(self, **kwargs) -> bytes:
        """Capture the current screen for vision observation."""
        import pyautogui
        from io import BytesIO
        # Note: In a real environment, we might want to target the Chrome window 
        # but for simplicity, we capture the full screen.
        img = pyautogui.screenshot()
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @property
    def url(self) -> str:
        return self._url

    def is_closed(self) -> bool:
        return self._is_closed

    async def close(self) -> None:
        self._is_closed = True

    @property
    def context(self) -> Any:
        return self

    @property
    def pages(self) -> list[Any]:
        return [self]

    async def bring_to_front(self) -> None:
        pass


class AsyncBrowserLauncher:
    """
    Simplified launcher that provides a HumanModeEmulator.
    
    All CDP/9222 logic has been removed to ensure stability and 
    compatibility with existing Chrome profiles.
    """

    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_env(cls) -> "AsyncBrowserLauncher":
        return cls()

    async def start(self) -> HumanModeEmulator:
        """
        Initialize the 'Human Mode' browser interface.
        """
        logger.info("Initializing Human Mode (Vision-Only)...")
        return HumanModeEmulator()

    async def stop(self) -> None:
        """No background processes to stop in pure Human Mode."""
        pass
