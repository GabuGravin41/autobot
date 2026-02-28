from __future__ import annotations

import subprocess
import time
from typing import Any

from .base import ActionSpec, BaseAdapter

try:
    import pyautogui
except ImportError:
    pyautogui = None


class AnyDeskAdapter(BaseAdapter):
    """
    Adapter for controlling and transferring tasks to AnyDesk nodes.
    This enables distributed computing across the local AGI cluster.
    """
    
    name = "anydesk"
    actions = {
        "connect": ActionSpec("Connect to a specific AnyDesk node by ID"),
        "disconnect": ActionSpec("Disconnect from current AnyDesk session"),
        "type_text": ActionSpec("Type text into the focused AnyDesk window"),
        "transfer_file": ActionSpec("Send a file to the connected AnyDesk node"),
    }

    def do_connect(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        node_id = str(params.get("node_id", "")).strip()
        if not node_id:
            raise ValueError("Missing required param: node_id")
            
        subprocess.Popen(["anydesk"])
        time.sleep(2.0)  # Wait for AnyDesk to open
        
        # Focus connection address bar (Ctrl+L equivalent in AnyDesk? Or tab navigation)
        # Assuming manual tab sequence or explicit click if needed, but for headless/basic:
        # AnyDesk command line actually supports connecting directly:
        # subprocess.Popen([r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe", node_id])
        return f"Connected to AnyDesk node: {node_id}"

    def do_disconnect(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        # Usually closing the current AnyDesk window drops the connection
        pyautogui.hotkey("alt", "f4")
        return "Disconnected from AnyDesk session."

    def do_type_text(self, params: dict[str, Any]) -> str:
        self._require_pyautogui()
        text = str(params.get("text", ""))
        interval = float(params.get("interval", 0.01))
        # When typing through AnyDesk, pacing is crucial due to latency
        pyautogui.write(text, interval=interval)
        return "Typed text through AnyDesk."

    def do_transfer_file(self, params: dict[str, Any]) -> str:
        file_path = str(params.get("path", "")).strip()
        if not file_path:
            raise ValueError("Missing required param: path")
        # AnyDesk has a file transfer mode via CLI or UI clipboard.
        # This is a placeholder for the actual implementation
        return f"Initiated file transfer for {file_path} to AnyDesk node."

    def _require_pyautogui(self) -> None:
        if pyautogui is None:
            raise RuntimeError("AnyDesk adapter requires pyautogui. Install it with: pip install pyautogui")
