"""
VS Code Tool — Programmatic orchestration of Visual Studio Code from Autobot.
Uses the official `code` CLI to open workspaces, files, diffs, and inspect installed extensions.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


class VSCode:
    """Autobot VS Code IDE Orchestration Tool."""

    def __init__(self, executable: str = "code"):
        self.executable = executable

    def is_available(self) -> bool:
        """Check if the `code` CLI is installed and discoverable on PATH."""
        return shutil.which(self.executable) is not None

    def open_workspace(self, workspace_path: str = ".") -> Dict[str, Any]:
        """Open a directory or workspace file in VS Code.
        
        Args:
            workspace_path: Path to directory or .code-workspace file.
        """
        p = Path(workspace_path).resolve()
        if not p.exists():
            return {"ok": False, "error": f"Path not found: {p}"}
        try:
            subprocess.Popen([self.executable, str(p)], shell=False)
            return {"ok": True, "opened": str(p)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def open_file(self, filepath: str, line: Optional[int] = None) -> Dict[str, Any]:
        """Open a specific file in VS Code, optionally jumping to a line number.
        
        Args:
            filepath: Path to the file.
            line: Optional line number to jump to.
        """
        p = Path(filepath).resolve()
        if not p.exists():
            return {"ok": False, "error": f"File not found: {p}"}
        try:
            target = f"{p}:{line}" if line else str(p)
            cmd = [self.executable, "-g", target] if line else [self.executable, str(p)]
            subprocess.Popen(cmd, shell=False)
            return {"ok": True, "opened": target}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def diff_files(self, file_left: str, file_right: str) -> Dict[str, Any]:
        """Open a side-by-side graphical diff in VS Code.
        
        Args:
            file_left: Original file path.
            file_right: Modified file path.
        """
        pl = Path(file_left).resolve()
        pr = Path(file_right).resolve()
        try:
            subprocess.Popen([self.executable, "-d", str(pl), str(pr)], shell=False)
            return {"ok": True, "diff": f"{pl} <-> {pr}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_extensions(self) -> Dict[str, Any]:
        """List all currently installed VS Code extensions."""
        if not self.is_available():
            return {"ok": False, "error": "VS Code CLI ('code') is not available on PATH."}
        try:
            proc = subprocess.run([self.executable, "--list-extensions"], capture_output=True, text=True, timeout=15)
            if proc.returncode == 0:
                exts = [e.strip() for e in proc.stdout.splitlines() if e.strip()]
                return {"ok": True, "extensions_count": len(exts), "extensions": exts}
            return {"ok": False, "error": proc.stderr}
        except Exception as e:
            return {"ok": False, "error": str(e)}
