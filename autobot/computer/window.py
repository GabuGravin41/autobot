"""
Window Control — OS-level window management and native UI interaction.

Uses Windows UI Automation (UIA) to interact with non-browser applications.
"""
from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, List, Optional

import uiautomation as auto  # Windows-only; callers must guard this import

from autobot.dom.native_extraction import NativeExtractionService

logger = logging.getLogger(__name__)

# Threads that have already had COM initialized, so we only pay for it once each.
_com_ready: set[int] = set()
_com_lock = threading.Lock()


def _ensure_com_initialized() -> None:
    """Initialize COM on the CALLING thread if it hasn't been already.

    UI Automation is a COM API and COM is initialized per-thread, not per
    process. Every method here is invoked through asyncio.to_thread() by the
    agent's dispatcher, which runs it on an arbitrary thread-pool worker that
    has never called CoInitialize — so without this, every single native-app
    call fails with:

        [WinError -2147221008] CoInitialize has not been called

    We deliberately do NOT pair this with CoUninitialize: thread-pool threads
    are reused across many calls, so initializing once per thread and leaving
    it is both correct and cheaper than bracketing every call.
    """
    tid = threading.get_ident()
    if tid in _com_ready:
        return
    try:
        import comtypes
        # STA (apartment-threaded) is what UI Automation expects.
        comtypes.CoInitialize()
    except OSError as e:
        # RPC_E_CHANGED_MODE means COM is already up on this thread in a
        # different mode — that's fine, UIA will still work.
        logger.debug(f"CoInitialize on thread {tid}: {e}")
    except Exception as e:
        logger.debug(f"CoInitialize unavailable on thread {tid}: {e}")
    with _com_lock:
        _com_ready.add(tid)


def _needs_com(func: Callable) -> Callable:
    """Ensure COM is live on this thread before touching UI Automation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ensure_com_initialized()
        return func(*args, **kwargs)
    return wrapper

class Window:
    """Control native Windows applications and UI elements."""

    def __init__(self, mouse: Any, keyboard: Any):
        self._native_service = NativeExtractionService()
        self.mouse = mouse
        self.keyboard = keyboard

    @_needs_com
    def list_all(self) -> List[str]:
        """List all top-level window titles."""
        return [w.Name for w in auto.GetRootControl().GetChildren() if w.Name]

    @_needs_com
    def focus(self, title_query: str) -> bool:
        """Focus a window containing the given title text."""
        window = auto.WindowControl(searchDepth=1, Name=title_query)
        if window.Exists(0):
            window.SetFocus()
            window.SetActive()
            return True
        return False

    @_needs_com
    def active_title(self) -> str:
        """Return the title of the currently focused window.

        Used by the agent loop to decide whether it is looking at the browser
        (in which case the DOM snapshot already describes the screen) or at a
        native app (in which case it needs extract_ui()).
        """
        try:
            window = auto.GetForegroundWindow()
            if isinstance(window, int):
                window = auto.ControlFromHandle(window)
            return getattr(window, "Name", "") or ""
        except Exception as e:
            logger.debug(f"active_title failed: {e}")
            return ""

    @_needs_com
    def extract_ui(self) -> str:
        """Extract the UI tree of the currently focused native window.

        Element indices in the returned tree are the ones click()/type() below
        resolve against — they share this instance's selector map, so always
        extract and act through the same Window object rather than creating a
        separate NativeExtractionService.
        """
        return self._native_service.extract_active_window()

    @_needs_com
    def click(self, index: int) -> bool:
        """Click a native UI element by its index from the last extraction."""
        control = self._native_service.get_element_by_index(index)
        if control:
            try:
                # Try UIA's specific pattern first (most robust)
                if hasattr(control, "Invoke"):
                    control.Invoke()
                else:
                    control.Click(simulateMove=True)
                return True
            except Exception as e:
                logger.warning(f"Native click on [{index}] failed: {e}")
                # Fallback to coordinate-based click if needed
                rect = control.BoundingRectangle
                if rect:
                    cx = rect.left + (rect.right - rect.left) // 2
                    cy = rect.top + (rect.bottom - rect.top) // 2
                    self.mouse.click(cx, cy)
                    return True
        return False

    @_needs_com
    def type(self, index: int, text: str) -> bool:
        """Type text into a native UI element by its index."""
        control = self._native_service.get_element_by_index(index)
        if control:
            try:
                if hasattr(control, "GetValuePattern"):
                    control.GetValuePattern().SetValue(text)
                else:
                    control.SetFocus()
                    self.keyboard.type(text)
                return True
            except Exception as e:
                logger.warning(f"Native type into [{index}] failed: {e}")
        return False
