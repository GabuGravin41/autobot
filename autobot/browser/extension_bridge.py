"""
Extension Bridge — lets Python (CoreLoop, or any other caller) send a DOM
command to the Autobot Chrome extension and block for the result, without
CDP, without launching a new Chrome process, and without touching Chrome's
profile files at all.

Why this exists: CDP automation (the old design) has to attach to Chrome's
debug port or launch an isolated profile, which on Windows repeatedly hit
SingletonLock stalls and profile corruption (see ROADMAP.md / DESIGN_
PHILOSOPHY.md). The Autobot Chrome extension is already installed in the
user's real, already-logged-in Chrome with a content script running on
every page — it can read and click the DOM directly through permissions
Chrome already granted it. This module is the missing wire between "the
backend wants something done in the page" and "the extension actually does
it in the user's real tab."

Flow:
  1. A caller (CoreLoop, a test script, curl) calls bridge.run(cmd_type,
     **params) or hits POST /api/extension/command (web/app.py), which
     calls bridge.run() under the hood.
  2. run() enqueues a _Command and blocks on a threading.Event.
  3. The extension's background.js polls GET /api/extension/poll (roughly
     every second) and, when a command is waiting, forwards it to the
     active tab's content script via chrome.tabs.sendMessage.
  4. content.js executes it (read_text / list_elements / click_index /
     type_index — see content.js's runDomCommand) and replies.
  5. background.js POSTs the result to /api/extension/result, which calls
     bridge.complete(), which sets the Event and wakes the blocked run().

Everything here is in-memory and per-process: if the backend restarts, any
in-flight command is simply abandoned (the caller's run() times out).
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any


class _Command:
    __slots__ = ("id", "type", "params", "created", "event", "result")

    def __init__(self, cmd_id: str, cmd_type: str, params: dict[str, Any]) -> None:
        self.id = cmd_id
        self.type = cmd_type
        self.params = params
        self.created = time.time()
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None


class ExtensionBridge:
    """Thread-safe in-memory queue bridging HTTP polling to blocking Python callers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque[_Command] = deque()
        self._inflight: dict[str, _Command] = {}
        self._last_poll: float = 0.0   # last time the extension polled (liveness signal)

    # ── Called on behalf of the extension (via web/app.py) ──────────────────

    def take_next(self) -> dict[str, Any] | None:
        """
        The extension's background.js polls this. Returns the next queued
        command (and marks it in-flight so a second poll doesn't grab it
        too), or None if nothing is waiting.
        """
        self._last_poll = time.time()
        with self._lock:
            if not self._queue:
                return None
            cmd = self._queue.popleft()
            self._inflight[cmd.id] = cmd
            return {"id": cmd.id, "type": cmd.type, "params": cmd.params}

    def complete(self, command_id: str, ok: bool, data: Any = None, error: str = "") -> bool:
        """The extension posts the executed result here. Returns False if the
        command already timed out and was discarded by run()."""
        with self._lock:
            cmd = self._inflight.pop(command_id, None)
        if cmd is None:
            return False
        cmd.result = {"ok": ok, "data": data, "error": error}
        cmd.event.set()
        return True

    @property
    def extension_connected(self) -> bool:
        """True if the extension has polled in the last 10 seconds — i.e. it's
        installed, enabled, and pointed at this backend right now."""
        return (time.time() - self._last_poll) < 10

    # ── Called by CoreLoop / any Python caller ───────────────────────────────

    def run(self, cmd_type: str, timeout: float = 10.0, **params: Any) -> dict[str, Any]:
        """
        Submit a command and block (this thread only — call via
        asyncio.to_thread from async code) until the extension answers or
        the timeout elapses.

        Returns {"ok": bool, "data": ..., "error": ...}. Never raises —
        a timeout or extension-side failure comes back as ok=False with a
        human-readable error, so callers (including the LLM loop) can just
        read the result.
        """
        cmd = _Command(uuid.uuid4().hex[:8], cmd_type, dict(params))
        with self._lock:
            self._queue.append(cmd)

        got_response = cmd.event.wait(timeout)

        with self._lock:
            self._inflight.pop(cmd.id, None)
            try:
                self._queue.remove(cmd)
            except ValueError:
                pass  # already taken off the queue by take_next()

        if not got_response:
            hint = (
                "extension not connected — is it installed, enabled, and pointed "
                "at this backend's URL?" if not self.extension_connected else
                "extension is connected but didn't answer in time — check the "
                "active tab isn't a chrome:// page, and that a normal tab is focused."
            )
            return {
                "ok": False,
                "data": None,
                "error": f"Timed out after {timeout}s waiting for the extension ({hint})",
            }
        return cmd.result or {"ok": False, "data": None, "error": "internal error: no result recorded"}


# One bridge per backend process, shared by web/app.py's HTTP endpoints and
# CoreLoop's browser_* actions.
bridge = ExtensionBridge()
