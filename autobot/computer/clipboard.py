"""
Clipboard Control — Read and write to the system clipboard.

Adapted from Open Interpreter's computer/clipboard/clipboard.py.
"""
from __future__ import annotations

import logging
import platform
import time

logger = logging.getLogger(__name__)


class Clipboard:
    """Read and write the system clipboard."""

    def get(self) -> str:
        """Get the current clipboard contents.

        Returns:
            The text currently in the clipboard.
        """
        try:
            import pyperclip
            content = pyperclip.paste()
        except ImportError:
            content = self._fallback_get()
        logger.debug(f"Clipboard get: '{content[:80]}...'" if len(content) > 80 else f"Clipboard get: '{content}'")
        return content

    def set(self, text: str) -> None:
        """Set the clipboard contents.

        Args:
            text: The text to copy to the clipboard.
        """
        try:
            import pyperclip
            pyperclip.copy(text)
        except ImportError:
            self._fallback_set(text)
        logger.debug(f"Clipboard set: '{text[:80]}...'" if len(text) > 80 else f"Clipboard set: '{text}'")

    def copy(self) -> str:
        """Press Ctrl+C and return the clipboard contents.

        Waits up to 0.5s for the clipboard to update, with retry logic.

        Returns:
            The text that was copied.
        """
        import pyautogui

        # Get clipboard before copy to detect change
        old_content = ""
        try:
            old_content = self.get()
        except Exception:
            pass

        pyautogui.hotkey("ctrl", "c")

        # Wait for clipboard to update (up to 0.5s, checking every 0.1s)
        for _ in range(5):
            time.sleep(0.1)
            try:
                new_content = self.get()
                if new_content != old_content:
                    return new_content
            except Exception:
                continue

        # Clipboard didn't change — return whatever is there
        return self.get()

    def paste(self) -> None:
        """Press Ctrl+V to paste clipboard contents."""
        import pyautogui
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.05)  # Brief pause for paste to register
        logger.debug("Clipboard paste (Ctrl+V)")

    def set_from_file(self, path: str, prefix: str = "", suffix: str = "") -> str:
        """Read a file and copy its contents (optionally with prefix/suffix text) to the clipboard.

        This is the correct single-step way to load a file's content onto the clipboard.
        Use this instead of trying to nest computer.files.read() inside clipboard.set().

        Args:
            path:   Path to the file. Supports ~ expansion.
            prefix: Optional text to prepend before the file content (e.g. a prompt).
            suffix: Optional text to append after the file content.

        Returns:
            Summary string: 'copied N chars from path/to/file to clipboard'

        Example:
            computer.clipboard.set_from_file('~/Desktop/paper.txt')
            computer.clipboard.set_from_file('~/Desktop/paper.txt', prefix='Review this paper:\\n\\n')
        """
        from pathlib import Path
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"File not found: {p}"
            content = p.read_text(encoding="utf-8", errors="replace")
            full_text = prefix + content + suffix
            self.set(full_text)
            logger.info(f"clipboard.set_from_file: copied {len(full_text)} chars from {p}")
            return f"copied {len(full_text)} chars from {p} to clipboard"
        except Exception as e:
            logger.warning(f"clipboard.set_from_file({path!r}) failed: {e}")
            return f"error: {e}"

    def prepend(self, text: str) -> str:
        """Prepend text to whatever is currently on the clipboard.

        Use this after clipboard.set_from_file() to add a prompt before the file content.
        Avoids the need to nest calls or reference variables.

        Args:
            text: Text to add BEFORE the current clipboard contents.

        Returns:
            'prepended N chars — clipboard now has M chars total'

        Example:
            computer.clipboard.set_from_file('~/Desktop/paper.txt')
            computer.clipboard.prepend('Please review this paper:\\n\\n')
        """
        try:
            current = self.get()
            new_text = text + current
            self.set(new_text)
            logger.info(f"clipboard.prepend: {len(text)} chars prepended, total {len(new_text)} chars")
            return f"prepended {len(text)} chars — clipboard now has {len(new_text)} chars total"
        except Exception as e:
            return f"error: {e}"

    def append(self, text: str) -> str:
        """Append text to whatever is currently on the clipboard.

        Args:
            text: Text to add AFTER the current clipboard contents.

        Returns:
            'appended N chars — clipboard now has M chars total'
        """
        try:
            current = self.get()
            new_text = current + text
            self.set(new_text)
            logger.info(f"clipboard.append: {len(text)} chars appended, total {len(new_text)} chars")
            return f"appended {len(text)} chars — clipboard now has {len(new_text)} chars total"
        except Exception as e:
            return f"error: {e}"

    def _fallback_get(self) -> str:
        """Platform-specific fallback for reading clipboard."""
        import subprocess
        system = platform.system()
        try:
            if system == "Linux":
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0:
                    return result.stdout
                # Try xsel as secondary fallback
                result = subprocess.run(
                    ["xsel", "--clipboard", "--output"],
                    capture_output=True, text=True, timeout=2,
                )
                return result.stdout
            elif system == "Darwin":
                result = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, timeout=2,
                )
                return result.stdout
            else:  # Windows
                result = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=2,
                )
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Clipboard fallback get failed: {e}")
            return ""

    def _fallback_set(self, text: str) -> None:
        """Platform-specific fallback for writing clipboard."""
        import subprocess
        system = platform.system()
        try:
            if system == "Linux":
                proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                )
                proc.communicate(text.encode())
            elif system == "Darwin":
                proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                proc.communicate(text.encode())
            else:  # Windows
                subprocess.run(
                    ["powershell", "-command", f"Set-Clipboard -Value '{text}'"],
                    capture_output=True, timeout=2,
                )
        except Exception as e:
            logger.warning(f"Clipboard fallback set failed: {e}")
