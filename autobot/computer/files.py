"""
Files Tool — File system navigation for Autobot.

Gives the agent structured access to the file system without needing to
open a file manager and take screenshots. Fast, accurate, token-efficient.

Usage (agent computer_call):
    computer.files.list(path="~/projects")
    computer.files.read(path="~/projects/main.py")
    computer.files.search(name="*.py", under="~/projects")
    computer.files.recent(path="~/Desktop", n=10)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_READ_BYTES = 32_000   # ~8k tokens — enough for most files
_MAX_LIST_ENTRIES = 100


class Files:
    """
    File system navigation tool.

    Methods:
        list(path, show_hidden=False)          — List files and folders at path
        read(path, lines=200)                  — Read a text file (first N lines)
        search(name, under="~", max_results=20) — Find files matching a glob pattern
        recent(path="~", n=10)                 — List N most recently modified files
        exists(path)                           — Check if a file or folder exists
        tree(path, depth=2)                    — Show directory tree (like `tree` command)
    """

    # ── list ──────────────────────────────────────────────────────────────────

    def list(self, path: str = "~", show_hidden: bool = False) -> str:
        """
        List files and folders at the given path.

        Returns a formatted directory listing with sizes, types and dates.
        Use this instead of opening a file manager.

        Args:
            path:        Directory to list. Supports ~ expansion.
            show_hidden: Include hidden files (starting with .). Default False.

        Returns:
            Formatted listing string.
        """
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"Path does not exist: {p}"
            if not p.is_dir():
                return f"Not a directory: {p} — use files.read() to read a file."

            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            lines = [f"📁 {p}\n"]
            count = 0
            for entry in entries:
                if not show_hidden and entry.name.startswith("."):
                    continue
                if count >= _MAX_LIST_ENTRIES:
                    lines.append(f"  ... (truncated, {len(list(p.iterdir()))} total)")
                    break
                try:
                    stat = entry.stat()
                    size = _fmt_size(stat.st_size) if entry.is_file() else ""
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    icon = "📄" if entry.is_file() else "📂"
                    lines.append(f"  {icon} {entry.name:<40} {size:<8} {mtime}")
                    count += 1
                except PermissionError:
                    lines.append(f"  🔒 {entry.name}  (permission denied)")
            if count == 0:
                lines.append("  (empty directory)")
            return "\n".join(lines)
        except Exception as e:
            return f"files.list error: {e}"

    # ── read ──────────────────────────────────────────────────────────────────

    def read(self, path: str, lines: int = 200) -> str:
        """
        Read a text file and return its contents.

        Args:
            path:  Path to the file. Supports ~ expansion.
            lines: Maximum number of lines to return (default 200).

        Returns:
            File contents as string, or an error message.
        """
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"File not found: {p}"
            if p.is_dir():
                return f"That's a directory, not a file. Use files.list('{path}') instead."
            if p.stat().st_size > _MAX_READ_BYTES * 4:
                return (
                    f"File is large ({_fmt_size(p.stat().st_size)}). "
                    f"Reading first {lines} lines only.\n\n"
                    + _read_lines(p, lines)
                )
            content = _read_lines(p, lines)
            total_lines = content.count("\n") + 1
            header = f"📄 {p}  ({_fmt_size(p.stat().st_size)}, {total_lines} lines)\n{'─' * 60}\n"
            return header + content
        except Exception as e:
            return f"files.read error: {e}"

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, name: str, under: str = "~", max_results: int = 20) -> str:
        """
        Find files matching a glob pattern under a directory.

        Args:
            name:        Glob pattern, e.g. "*.py", "main*", "report*.pdf"
            under:       Root directory to search from. Default: home directory.
            max_results: Maximum number of results to return.

        Returns:
            List of matching file paths.
        """
        try:
            root = Path(under).expanduser().resolve()
            if not root.exists():
                return f"Directory not found: {root}"
            matches = []
            for match in root.rglob(name):
                if match.name.startswith("."):
                    continue  # skip hidden
                try:
                    stat = match.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    size = _fmt_size(stat.st_size) if match.is_file() else "dir"
                    matches.append((mtime, f"  {match}  ({size})  {mtime}"))
                except Exception:
                    matches.append(("", f"  {match}"))
                if len(matches) >= max_results:
                    break
            if not matches:
                return f"No files matching '{name}' found under {root}"
            matches.sort(reverse=True)
            lines = [f"Search '{name}' under {root}:\n"]
            lines.extend(m[1] for m in matches[:max_results])
            return "\n".join(lines)
        except Exception as e:
            return f"files.search error: {e}"

    # ── recent ────────────────────────────────────────────────────────────────

    def recent(self, path: str = "~", n: int = 10) -> str:
        """
        List the N most recently modified files in a directory (non-recursive).

        Useful for finding where work was last saved.

        Args:
            path: Directory to check. Default: home directory.
            n:    Number of recent files to return.

        Returns:
            Formatted list of recent files with modification times.
        """
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"Path not found: {p}"
            files = [e for e in p.iterdir() if e.is_file() and not e.name.startswith(".")]
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            lines = [f"Most recent files in {p}:\n"]
            for f in files[:n]:
                stat = f.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"  {f.name:<45} {_fmt_size(stat.st_size):<8} {mtime}")
            return "\n".join(lines) if len(lines) > 1 else f"No files found in {p}"
        except Exception as e:
            return f"files.recent error: {e}"

    # ── exists ────────────────────────────────────────────────────────────────

    def exists(self, path: str) -> str:
        """
        Check whether a file or folder exists at the given path.

        Args:
            path: Path to check. Supports ~ expansion.

        Returns:
            Short status string: "exists (file)", "exists (dir)", or "does not exist".
        """
        p = Path(path).expanduser().resolve()
        if p.is_file():
            return f"exists (file, {_fmt_size(p.stat().st_size)}): {p}"
        if p.is_dir():
            return f"exists (directory): {p}"
        return f"does not exist: {p}"

    # ── tree ──────────────────────────────────────────────────────────────────

    def tree(self, path: str = ".", depth: int = 2) -> str:
        """
        Show a directory tree (like the `tree` command).

        Args:
            path:  Root of the tree. Default: current directory.
            depth: How many levels deep to show (default 2, max 4).

        Returns:
            Indented tree string.
        """
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"Path not found: {p}"
            depth = min(depth, 4)
            lines = [str(p)]
            _build_tree(p, lines, prefix="", current_depth=0, max_depth=depth)
            return "\n".join(lines)
        except Exception as e:
            return f"files.tree error: {e}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size //= 1024
    return f"{size}GB"


def _read_lines(p: Path, max_lines: int) -> str:
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            content = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    content.append(f"\n... (file continues — use lines={max_lines * 2} to see more)")
                    break
                content.append(line)
            return "".join(content)
    except Exception as e:
        return f"(read error: {e})"


def _build_tree(path: Path, lines: list, prefix: str, current_depth: int, max_depth: int) -> None:
    if current_depth >= max_depth:
        return
    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        entries = [e for e in entries if not e.name.startswith(".")]
        for i, entry in enumerate(entries[:50]):  # cap at 50 per level
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                _build_tree(entry, lines, prefix + extension, current_depth + 1, max_depth)
    except PermissionError:
        lines.append(f"{prefix}[permission denied]")
