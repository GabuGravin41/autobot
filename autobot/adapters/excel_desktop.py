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


class ExcelDesktopAdapter(BaseAdapter):
    """Adapter for basic offline Excel tasks using UI automation."""
    name = "excel_desktop"
    actions = {
        "open_excel": ActionSpec("Open Microsoft Excel"),
        "open_workbook": ActionSpec("Open an existing Excel workbook"),
        "write_cell": ActionSpec("Write text to the currently active cell and press Enter"),
        "save_workbook": ActionSpec("Save the current workbook"),
        "navigate_cells": ActionSpec("Navigate between cells using arrow keys"),
    }

    def do_open_excel(self, params: dict[str, Any]) -> str:
        # On Windows, 'excel' might launch it if it's in PATH or AppPaths, or use start excel
        subprocess.Popen(["start", "excel"], shell=True)
        time.sleep(3.0) # Wait for Excel to load
        return "Opened Microsoft Excel."

    def do_open_workbook(self, params: dict[str, Any]) -> str:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("Missing required param: path")
        file_path = Path(path).resolve()
        
        # Open file natively
        subprocess.Popen(["start", "", str(file_path)], shell=True)
        time.sleep(3.0)
        return f"Opened Excel workbook: {file_path}"

    def do_write_cell(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        text = str(params.get("text", ""))
        pyautogui.write(text, interval=0.01)
        pyautogui.press("enter")
        return f"Wrote '{text}' into current cell."

    def do_navigate_cells(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        direction = str(params.get("direction", "down")).lower()
        amount = int(params.get("amount", 1))
        
        keys = {"up": "up", "down": "down", "left": "left", "right": "right"}
        key = keys.get(direction, "down")
        
        for _ in range(amount):
            pyautogui.press(key)
            time.sleep(0.05)
            
        return f"Navigated {amount} cells {direction}."

    def do_save_workbook(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        pyautogui.hotkey("ctrl", "s")
        time.sleep(0.5)
        return "Saved Excel workbook."

    def _require_pyautogui(self) -> None:
        if pyautogui is None:
            raise RuntimeError("Excel desktop adapter requires pyautogui. Install it with: pip install pyautogui")
