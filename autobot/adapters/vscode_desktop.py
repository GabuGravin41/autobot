from __future__ import annotations

import subprocess
from typing import Any

try:
    import pyautogui
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None

from .base import ActionSpec, BaseAdapter


class VSCodeDesktopAdapter(BaseAdapter):
    name = "vscode_desktop"
    actions = {
        "open_vscode": ActionSpec("Open VS Code with optional path"),
        "open_file": ActionSpec("Open file in VS Code quick open"),
        "run_terminal_command": ActionSpec("Run command in VS Code integrated terminal"),
        "type_text": ActionSpec("Type text into active editor"),
        "save_file": ActionSpec("Save active file"),
    }

    def do_open_vscode(self, params: dict[str, Any]) -> str:
        path = str(params.get("path", "")).strip()
        if path:
            subprocess.Popen(["code", path])
            return f"Opened VS Code at: {path}"
        subprocess.Popen(["code"])
        return "Opened VS Code."

    def do_open_file(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        file_path = str(params.get("path", "")).strip()
        if not file_path:
            raise ValueError("Missing required param: path")
        pyautogui.hotkey("ctrl", "p")
        pyautogui.write(file_path, interval=0.01)
        pyautogui.press("enter")
        return f"Opened file in VS Code: {file_path}"

    def do_run_terminal_command(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        command = str(params.get("command", "")).strip()
        if not command:
            raise ValueError("Missing required param: command")
        pyautogui.hotkey("ctrl", "`")
        pyautogui.write(command, interval=0.01)
        pyautogui.press("enter")
        return f"Ran VS Code terminal command: {command}"

    def do_type_text(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        text = str(params.get("text", ""))
        interval = float(params.get("interval", 0.01))
        pyautogui.write(text, interval=interval)
        return "Typed text in VS Code."

    def do_save_file(self, _params: dict[str, Any]) -> str:
        self._require_pyautogui()
        pyautogui.hotkey("ctrl", "s")
        return "Saved active file."

    def _require_pyautogui(self) -> None:
        if pyautogui is None:
            raise RuntimeError("VS Code desktop adapter requires pyautogui. Install it with: pip install pyautogui")
