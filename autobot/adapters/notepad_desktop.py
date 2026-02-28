from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from .base import ActionSpec, BaseAdapter

try:
    import pyautogui
except ImportError:
    pyautogui = None


class NotepadDesktopAdapter(BaseAdapter):
    """Adapter for basic offline Notepad tasks using UI automation."""
    name = "notepad_desktop"
    actions = {
        "open_notepad": ActionSpec("Open Microsoft Notepad"),
        "open_file": ActionSpec("Open an existing text file"),
        "type_text": ActionSpec("Type text into the editor"),
        "save_file": ActionSpec("Save the current file"),
    }

    def do_open_notepad(self, params: dict[str, Any]) -> str:
        subprocess.Popen(["notepad.exe"])
        time.sleep(2.0)
        return "Opened Microsoft Notepad."

    def do_open_file(self, params: dict[str, Any]) -> str:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("Missing required param: path")
        file_path = Path(path).resolve()
        
        subprocess.Popen(["notepad.exe", str(file_path)])
        time.sleep(2.0)
        return f"Opened Notepad file: {file_path}"

    def do_type_text(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        text = str(params.get("text", ""))
        pyautogui.write(text, interval=0.01)
        return f"Typed '{text}' into Notepad."

    def do_save_file(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        pyautogui.hotkey("ctrl", "s")
        time.sleep(0.5)
        return "Saved Notepad file."

    def _require_pyautogui(self) -> None:
        if pyautogui is None:
            raise RuntimeError("Notepad desktop adapter requires pyautogui. Install it with: pip install pyautogui")
