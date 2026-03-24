"""
Display Control — Screenshots and screen information.

Adapted from Open Interpreter's computer/display/display.py.
Provides screenshot capture and screen dimension queries.
"""
from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)


class Display:
    """Capture screenshots and get screen information."""

    def screenshot(self) -> str:
        """Take a screenshot of the entire screen, returns base64-encoded PNG.

        Returns:
            Base64-encoded PNG string of the current screen.
        """
        import pyautogui
        from io import BytesIO
        img = pyautogui.screenshot()
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug("Screenshot captured")
        return b64

    def screenshot_region(self, x: int, y: int, width: int, height: int) -> str:
        """Take a screenshot of a specific screen region, returns base64 PNG.

        Args:
            x: Left coordinate of the region.
            y: Top coordinate of the region.
            width: Width of the region.
            height: Height of the region.

        Returns:
            Base64-encoded PNG of the specified region.
        """
        import pyautogui
        from io import BytesIO
        img = pyautogui.screenshot(region=(x, y, width, height))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug(f"Screenshot region ({x},{y},{width},{height}) captured")
        return b64

    @staticmethod
    def size() -> tuple[int, int]:
        """Get the screen resolution.

        Returns:
            Tuple of (width, height) in pixels.
        """
        import pyautogui
        s = pyautogui.size()
        return (s.width, s.height)

    @staticmethod
    def width() -> int:
        """Get screen width in pixels."""
        import pyautogui
        return pyautogui.size().width

    @staticmethod
    def height() -> int:
        """Get screen height in pixels."""
        import pyautogui
        return pyautogui.size().height

    @staticmethod
    def windows() -> str:
        """
        List all open windows with their titles and applications.

        Returns a formatted list so the agent knows what apps are open
        without needing to take a screenshot. Use this to orient yourself
        at the start of a task or to find which window to switch to.

        Returns:
            Formatted list of open windows, or a screenshot-based fallback.
        """
        import platform
        import subprocess

        system = platform.system()
        try:
            if system == "Linux":
                # wmctrl -l lists: id  desktop  host  title
                result = subprocess.run(
                    ["wmctrl", "-l"], capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    lines = ["Open windows:\n"]
                    for line in result.stdout.strip().splitlines():
                        parts = line.split(None, 3)
                        title = parts[3] if len(parts) > 3 else "(no title)"
                        # Strip the hostname from the title
                        import socket
                        hostname = socket.gethostname()
                        title = title.replace(hostname, "").strip()
                        if title and title not in ("Desktop",):
                            lines.append(f"  • {title}")
                    return "\n".join(lines) if len(lines) > 1 else "No windows open."

            elif system == "Darwin":  # macOS
                script = 'tell application "System Events" to get name of every process whose background only is false'
                result = subprocess.run(
                    ["osascript", "-e", script], capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    apps = [a.strip() for a in result.stdout.strip().split(",")]
                    return "Open applications:\n" + "\n".join(f"  • {a}" for a in apps if a)

            elif system == "Windows":
                import ctypes
                # Use EnumWindows via subprocess powershell
                ps = "Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object Name,MainWindowTitle | Format-Table -AutoSize"
                result = subprocess.run(
                    ["powershell", "-Command", ps],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return "Open windows:\n" + result.stdout.strip()

        except FileNotFoundError:
            return "Window listing unavailable (wmctrl not installed). Take a screenshot to see what's open."
        except Exception as e:
            logger.debug(f"windows() failed: {e}")

        return "Window listing unavailable on this platform. Use screenshot to see what's open."

    @staticmethod
    def window_titles() -> frozenset:
        """
        Return a frozenset of currently open window titles via wmctrl.

        Used for OS-dialog detection: diff window lists before/after an action
        to detect new system dialogs (file pickers, GTK dialogs, auth prompts)
        that are invisible to DOM-based popup detection.

        Returns an empty frozenset if wmctrl is unavailable (non-Linux or not installed).
        Each call takes ~5-10ms — negligible for per-action checks.
        """
        import platform
        import subprocess
        import socket

        if platform.system() != "Linux":
            return frozenset()
        try:
            result = subprocess.run(
                ["wmctrl", "-l"], capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return frozenset()
            hostname = socket.gethostname()
            titles: set[str] = set()
            for line in result.stdout.strip().splitlines():
                parts = line.split(None, 3)
                if len(parts) > 3:
                    title = parts[3].replace(hostname, "").strip()
                    if title and title not in ("Desktop",):
                        titles.add(title)
            return frozenset(titles)
        except Exception:
            return frozenset()

    @staticmethod
    def focus(window_title: str) -> str:
        """
        Bring a window to the foreground by searching its title.

        Args:
            window_title: Partial title of the window to focus (case-insensitive).

        Returns:
            Status string confirming which window was focused.
        """
        import platform
        import subprocess

        system = platform.system()
        try:
            if system == "Linux":
                result = subprocess.run(
                    ["wmctrl", "-a", window_title],
                    capture_output=True, text=True, timeout=3,
                )
                import time; time.sleep(0.3)
                if result.returncode == 0:
                    return f"Focused window matching: '{window_title}'"
                return f"No window found matching: '{window_title}'. Use display.windows() to see what's open."

            elif system == "Darwin":
                script = f'tell application "{window_title}" to activate'
                subprocess.run(["osascript", "-e", script], timeout=3)
                import time; time.sleep(0.3)
                return f"Activated: {window_title}"

        except Exception as e:
            return f"focus() failed: {e}. Try Alt+Tab instead."

        return f"focus() not supported on {system}."
