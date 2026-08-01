"""
ScreenLock — Mutual exclusion for mouse/keyboard access.

Only one agent can actively control the screen at a time.
When a task is waiting for an LLM response or sleeping, it releases the
lock so other queued tasks can take control of the computer.

Smooth handoff design:
    - Each task registers a "window hint" (e.g. "Google Chrome", "code") so that
      when it re-acquires the lock after another task used the screen, the lock
      automatically refocuses its window — exactly like a human Alt-Tabbing back.
    - last_released_by tracks the previous holder so AgentLoop knows whether a
      context switch occurred and needs a fresh focus + settle.
    - _waiters exposes the queue to the dashboard for live visibility.

Usage:
    # Register which window this task works in (call once, update on navigate)
    screen_lock.register_window("task-abc", "Google Chrome")

    async with screen_lock.acquire("task-abc", goal="clicking Submit"):
        # screen_lock.context_switched → True if another task acted in between
        if screen_lock.context_switched:
            focus_window_and_wait(...)
        computer.mouse.click(x, y)
    # lock released — waiter with highest priority gets it next
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_ACQUIRE_TIMEOUT = float("inf")   # wait forever by default (queue semantics)
_YIELD_GRACE_SECONDS = 0.15       # brief pause after release so waiter can start


class ScreenLock:
    """
    Asyncio-based mutual exclusion for the physical screen.

    Attributes:
        holder_id         : task_id currently holding the lock (or None)
        holder_goal       : short description of what the holder is doing
        acquired_at       : epoch seconds when lock was last acquired
        last_released_by  : task_id that most recently released the lock
        context_switched  : True if the last acquire was by a different task
                            than the previous holder — caller should refocus window
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.holder_id: str | None = None
        self.holder_goal: str = ""
        self.acquired_at: float = 0.0
        self.last_released_by: str | None = None   # updated on every release
        self.context_switched: bool = False        # set True when a different task acquires

        # Window hint registry: task_id → partial window title for display.focus()
        self._window_hints: dict[str, str] = {}

        # Waiter list (display only — not used for scheduling, asyncio.Lock is FIFO)
        # Each entry: {task_id, goal, waiting_since}
        self._waiters: list[dict] = []

    # ── Window hint API ────────────────────────────────────────────────────────

    def register_window(self, task_id: str, hint: str) -> None:
        """
        Track which window/app this task is working in.

        Called by AgentLoop when navigating or calling display.focus().
        Used to restore window focus on context switch.

        hint: partial window title understood by wmctrl / display.focus()
              e.g. "Google Chrome", "code", "Terminal", "Files"
        """
        if hint:
            self._window_hints[task_id] = hint
            logger.debug(f"Window hint registered: {task_id} → '{hint}'")

    def get_window_hint(self, task_id: str) -> str | None:
        """Return the last known window hint for a task (None if unknown)."""
        return self._window_hints.get(task_id)

    def unregister(self, task_id: str) -> None:
        """Remove hints for a finished task."""
        self._window_hints.pop(task_id, None)
        self._waiters = [w for w in self._waiters if w["task_id"] != task_id]

    # ── Core acquire ───────────────────────────────────────────────────────────

    @contextlib.asynccontextmanager
    async def acquire(
        self,
        task_id: str,
        goal: str = "",
        timeout: float = _ACQUIRE_TIMEOUT,
    ) -> AsyncIterator[None]:
        """
        Context manager that acquires the screen lock for `task_id`.

        Blocks until the current holder releases it (or timeout is reached).
        Releases automatically on exit — even if the body raises.

        After acquiring, check `screen_lock.context_switched` to know if the
        screen state changed (another task was acting since this task last held it).

        Example:
            screen_lock.register_window("abc123", "Google Chrome")

            async with screen_lock.acquire("abc123", "clicking Submit"):
                if screen_lock.context_switched:
                    await asyncio.to_thread(computer.display.focus, "Google Chrome")
                    await asyncio.sleep(0.3)
                computer.mouse.click(x, y)
        """
        # Add to waiter list for dashboard visibility
        waiter_entry = {
            "task_id": task_id,
            "goal": goal[:60],
            "waiting_since": time.time(),
        }
        self._waiters.append(waiter_entry)

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            self._waiters = [w for w in self._waiters if w["task_id"] != task_id]
            raise TimeoutError(
                f"Task {task_id} could not acquire the screen lock within {timeout}s"
            )

        # Lock acquired — remove from waiter list, promote to holder
        self._waiters = [w for w in self._waiters if w["task_id"] != task_id]

        # Detect context switch: was the previous holder a DIFFERENT task?
        self.context_switched = (
            self.last_released_by is not None and self.last_released_by != task_id
        )

        self.holder_id = task_id
        self.holder_goal = goal
        self.acquired_at = time.time()

        if self.context_switched:
            logger.info(
                f"🔄 Screen context switch: {self.last_released_by} → {task_id} "
                f"(window: '{self._window_hints.get(task_id, 'unknown')}')"
            )
        else:
            logger.debug(f"🔒 Screen lock acquired by {task_id}: {goal}")

        try:
            yield
        finally:
            self.last_released_by = task_id
            self.holder_id = None
            self.holder_goal = ""
            self._lock.release()
            logger.debug(f"🔓 Screen lock released by {task_id}")
            # Brief yield so the next waiter can start before we continue
            await asyncio.sleep(_YIELD_GRACE_SECONDS)

    # ── Status ─────────────────────────────────────────────────────────────────

    def is_locked(self) -> bool:
        return self._lock.locked()

    def get_status(self) -> dict:
        return {
            "locked": self.is_locked(),
            "holder_id": self.holder_id,
            "holder_goal": self.holder_goal,
            "held_for_seconds": int(time.time() - self.acquired_at) if self.holder_id else 0,
            "last_released_by": self.last_released_by,
            "waiting_tasks": [
                {
                    "task_id": w["task_id"],
                    "goal": w["goal"],
                    "waiting_seconds": int(time.time() - w["waiting_since"]),
                }
                for w in self._waiters
            ],
        }


# ── Global singleton ──────────────────────────────────────────────────────────
# Import this in any module that needs screen access:
#   from autobot.agent.resource_manager import screen_lock

screen_lock = ScreenLock()
