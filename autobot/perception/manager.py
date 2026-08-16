"""
Perception Manager — Unified State Snapshot Engine.
Aggregates screen vision, browser DOM tree, active OS windows, and CLI status.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autobot.perception.screen import ScreenPerception

logger = logging.getLogger(__name__)


@dataclass
class PerceptionSnapshot:
    """Complete snapshot of laptop perception at step T."""
    screen_b64: str = ""
    screen_bytes: bytes = b""
    active_window_title: str = ""
    dom_snapshot: Optional[Any] = None
    browser_url: str = ""
    browser_title: str = ""
    open_windows: List[str] = field(default_factory=list)
    cli_output: str = ""

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert perception into a JSON-serializable context for LLM prompt."""
        return {
            "active_window": self.active_window_title,
            "browser_url": self.browser_url,
            "browser_title": self.browser_title,
            "open_windows": self.open_windows[:5],
            "has_vision": bool(self.screen_b64),
        }


class PerceptionManager:
    """Coordinates multi-modal perception across Screen, DOM, and OS Windows."""

    def __init__(self, cdp_port: int = 9222):
        self.cdp_port = cdp_port

    async def capture(self, page: Optional[Any] = None) -> PerceptionSnapshot:
        """Capture full perception snapshot."""
        snapshot = PerceptionSnapshot()

        # 1. Screen Vision
        raw_bytes, b64_str = ScreenPerception.capture_screenshot()
        snapshot.screen_bytes = raw_bytes
        snapshot.screen_b64 = b64_str

        # 2. Active Window & OS Info
        snapshot.active_window_title = self._get_active_window_title()
        snapshot.open_windows = self._get_open_windows()

        # 3. Browser State (if Playwright page object is active and open)
        if page and hasattr(page, "is_closed") and not page.is_closed():
            try:
                snapshot.browser_url = page.url
                snapshot.browser_title = await page.title()
            except Exception as e:
                logger.debug(f"Could not read browser page details: {e}")

        return snapshot

    @staticmethod
    def _get_active_window_title() -> str:
        """Get the title of the foreground active window on Windows/OS."""
        if os.name == "nt":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                return buf.value or "Desktop"
            except Exception:
                pass
        return "Unknown"

    @staticmethod
    def _get_open_windows() -> List[str]:
        """List open top-level application window titles."""
        if os.name == "nt":
            try:
                cmd = 'powershell "Get-Process | Where-Object {$_.MainWindowTitle -ne \'\'} | Select-Object -ExpandProperty MainWindowTitle"'
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
                if res.stdout:
                    return [w.strip() for w in res.stdout.splitlines() if w.strip()]
            except Exception:
                pass
        return []
