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

No `computer.browser` submodule (CDP retired, Sep 2026): the project
moved off Chrome DevTools Protocol entirely — it required a debug-port
Chrome launch that fought Windows SingletonLock and needed its own
isolated profile, disconnected from whatever Chrome the user actually
had open. Actual web page content now goes through the Chrome extension's
DOM bridge instead (autobot/browser/extension_bridge.py, the
browser_text/browser_list/browser_click/browser_type/browser_paste
CoreLoop actions) — no debug port, rides the user's real logged-in
Chrome. The old CDP-based `computer/browser.py` (Browser.url(),
click_element(), fill(), all built on autobot/dom/page_snapshot.py's
websocket client to a `--remote-debugging-port` Chrome that this
codebase no longer launches) is kept only under `_archive/` for
reference; it is deliberately NOT imported or instantiated here.

This matters beyond just avoiding a crash: get_tool_catalog() below
auto-lists every public method on every submodule attached to this
class. A retired, non-functional submodule wired in here doesn't just
sit unused — it actively appears in the LLM's tool catalog every step,
indistinguishable from a tool that actually works, and gets called.
That's exactly what happened before this fix: `computer.browser.url()`
still showed up in the catalog and a live run reached for it, got a
silent blank/empty result (no debug-port Chrome to answer), and the
model had no signal that the tool itself — not its own reasoning — was
the problem. The fix is structural, not a one-off: only attach a
submodule here once its live perception/action path is confirmed
working, and remove it from here the same day its replacement ships,
not "eventually."
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
from autobot.computer.claude_code_tool import ClaudeCode
from autobot.computer.antigravity_tool import Antigravity
from autobot.computer.research_tool import Research
from autobot.computer.vault import Vault
from autobot.computer.anti_sleep import anti_sleep
from autobot.computer.cleaner_tool import Cleaner
from autobot.computer.quant_tool import Quant
from autobot.computer.vscode_tool import VSCode
from autobot.computer.eris_tool import Eris

logger = logging.getLogger(__name__)

# Windows native-app control (UIA) is optional. `uiautomation` is a Windows-only
# COM package and is commented out in requirements.txt, so importing it
# unconditionally here made `Computer()` — and therefore CoreLoop, and
# therefore every entry point — fail at import on any Windows machine that
# installed from requirements.txt. Degrade to mouse/keyboard/browser-bridge
# control instead of taking the whole agent down with us.
Window = None  # type: ignore[assignment]
HAS_NATIVE_UI = False

if platform.system() == 'Windows':
    try:
        from autobot.computer.window import Window  # type: ignore[assignment]
        HAS_NATIVE_UI = True
    except ImportError as e:
        logger.warning(
            "Native Windows UI control unavailable (%s). Browser-bridge and "
            "mouse/keyboard control still work; native desktop apps (Artemis, "
            "VESTA, Excel...) do not. Install with: pip install uiautomation",
            e,
        )


class Computer:
    """
    Central computer control API.

    Adapted from Open Interpreter's Computer class. Provides a clean,
    documented API for OS-level automation: mouse, keyboard, display, clipboard.

    The key method is get_tool_catalog(), which auto-extracts method signatures
    and docstrings from all submodules and formats them for LLM injection.

    Deliberately no `self.browser` — see the module docstring above.
    """

    def __init__(self) -> None:
        self.mouse = Mouse()
        self.keyboard = Keyboard()
        self.display = Display()
        self.clipboard = Clipboard()
        self.files = Files()
        self.terminal = Terminal()
        self.kaggle = Kaggle()
        self.claude_code = ClaudeCode()
        self.antigravity = Antigravity()
        self.research = Research()
        self.vault = Vault()
        self.anti_sleep = anti_sleep
        self.cleaner = Cleaner()
        self.quant = Quant()
        self.vscode = VSCode()
        self.eris = Eris()
        if HAS_NATIVE_UI and Window is not None:
            self.window = Window(self.mouse, self.keyboard)

    def _get_all_tools(self) -> list[tuple[str, Any]]:
        names = ["mouse", "keyboard", "display", "clipboard",
                 "files", "terminal", "vault", "kaggle", "claude_code",
                 "antigravity", "research", "anti_sleep",
                 "cleaner", "quant", "vscode", "eris"]
        if hasattr(self, "window"):
            names.append("window")
        return [(name, getattr(self, name)) for name in names]

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

        for tool_name, tool in self._get_all_tools():
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
