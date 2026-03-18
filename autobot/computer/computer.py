"""
Computer API — Central class for OS-level computer control.

Adapted from Open Interpreter's computer/computer.py.
Aggregates all submodules (mouse, keyboard, display, clipboard)
and auto-generates a tool catalog from their docstrings.

The tool catalog is injected into the LLM's system prompt so it
always knows what OS-level tools are available and how to call them.

Usage:
    computer = Computer()
    computer.mouse.click(100, 200)
    computer.keyboard.type("hello")
    catalog = computer.get_tool_catalog()  # → inject into system prompt
"""
from __future__ import annotations

import inspect
import logging
import platform
from typing import Any

from autobot.computer.mouse import Mouse
from autobot.computer.keyboard import Keyboard
from autobot.computer.display import Display
from autobot.computer.clipboard import Clipboard
from autobot.computer.files import Files
from autobot.computer.terminal import Terminal
from autobot.computer.kaggle_tool import Kaggle
from autobot.computer.research_tool import Research
from autobot.computer.vault import Vault
from autobot.computer.anti_sleep import anti_sleep

if platform.system() == 'Windows':
    from autobot.computer.window import Window
    import uiautomation as auto

logger = logging.getLogger(__name__)


class Computer:
    """
    Central computer control API.

    Adapted from Open Interpreter's Computer class. Provides a clean,
    documented API for OS-level automation: mouse, keyboard, display, clipboard.

    The key method is get_tool_catalog(), which auto-extracts method signatures
    and docstrings from all submodules and formats them for LLM injection.
    """

    def __init__(self) -> None:
        self.mouse = Mouse()
        self.keyboard = Keyboard()
        self.display = Display()
        self.clipboard = Clipboard()
        self.files = Files()
        self.terminal = Terminal()
        self.kaggle = Kaggle()
        self.research = Research()
        self.vault = Vault()
        self.anti_sleep = anti_sleep
        if platform.system() == 'Windows':
            self.window = Window(self.mouse, self.keyboard)

    def _get_all_tools(self) -> list[Any]:
        """Get all tool submodules."""
        tools = [
            self.mouse,
            self.keyboard,
            self.display,
            self.clipboard,
            self.files,
            self.terminal,
            self.vault,
            self.kaggle,
            self.research,
            self.anti_sleep,
        ]
        if hasattr(self, 'window'):
            tools.append(self.window)
        return tools

    def get_tool_catalog(self) -> str:
        """
        Auto-generate a tool catalog by introspecting all submodules.

        Adapted from Open Interpreter's _get_all_computer_tools_signature_and_description().
        This extracts method signatures and docstrings and formats them for the LLM.

        Returns a string like:
            ## OS Control Tools (via computer module)
            computer.mouse.click(x, y, button='left', clicks=1) — Click at screen coordinates (x, y).
            computer.keyboard.type(text, interval=0.03) — Type text character by character.
            computer.display.screenshot() — Take a screenshot, returns base64 PNG.
            computer.clipboard.get() — Get clipboard contents.
        """
        lines: list[str] = ["## OS Control Tools (via computer module)"]
        lines.append("These tools control the physical computer — use for OS-level tasks outside the browser.")
        lines.append("")

        for tool in self._get_all_tools():
            tool_name = tool.__class__.__name__.lower()
            tool_methods = self._extract_methods(tool, tool_name)
            for method_info in tool_methods:
                lines.append(
                    f"- `computer.{method_info['signature']}` — {method_info['description']}"
                )

        return "\n".join(lines)

    def _extract_methods(self, tool: Any, tool_name: str) -> list[dict[str, str]]:
        """
        Extract method signatures and descriptions from a tool submodule.

        Adapted from Open Interpreter's _extract_tool_info() method.
        """
        methods: list[dict[str, str]] = []

        for name, method in inspect.getmembers(tool, predicate=lambda m: inspect.ismethod(m) or inspect.isfunction(m) or isinstance(m, staticmethod)):
            # Skip private/dunder methods
            if name.startswith("_"):
                continue

            # Get method signature
            try:
                sig = inspect.signature(method)
                params = []
                for param_name, param in sig.parameters.items():
                    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        params.append(f"*{param_name}")
                    elif param.default == param.empty:
                        params.append(param_name)
                    else:
                        params.append(f"{param_name}={param.default!r}")

                signature = f"{tool_name}.{name}({', '.join(params)})"
            except (ValueError, TypeError):
                signature = f"{tool_name}.{name}()"

            # Get first line of docstring
            doc = method.__doc__ or ""
            description = doc.strip().split("\n")[0] if doc else "No description available."

            methods.append({
                "signature": signature,
                "description": description,
            })
        return methods
