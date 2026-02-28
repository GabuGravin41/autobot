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


class WordDesktopAdapter(BaseAdapter):
    """Adapter for basic offline Word tasks using UI automation."""
    name = "word_desktop"
    actions = {
        "open_word": ActionSpec("Open Microsoft Word"),
        "open_document": ActionSpec("Open an existing Word document"),
        "type_text": ActionSpec("Type text into the document"),
        "save_document": ActionSpec("Save the current document"),
    }

    def do_open_word(self, params: dict[str, Any]) -> str:
        subprocess.Popen(["start", "winword"], shell=True)
        time.sleep(3.0) 
        return "Opened Microsoft Word."

    def do_open_document(self, params: dict[str, Any]) -> str:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("Missing required param: path")
        file_path = Path(path).resolve()
        
        subprocess.Popen(["start", "", str(file_path)], shell=True)
        time.sleep(3.0)
        return f"Opened Word document: {file_path}"

    def do_type_text(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        text = str(params.get("text", ""))
        pyautogui.write(text, interval=0.01)
        return f"Typed '{text}' into Word."

    def do_save_document(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        pyautogui.hotkey("ctrl", "s")
        time.sleep(0.5)
        return "Saved Word document."

    def _require_pyautogui(self) -> None:
        if pyautogui is None:
            raise RuntimeError("Word desktop adapter requires pyautogui. Install it with: pip install pyautogui")
