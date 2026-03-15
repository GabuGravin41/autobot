"""
ScreenLock — Mutual exclusion for mouse/keyboard access.

Only one agent can actively control the screen at a time.
When a task is waiting for an LLM response or sleeping, it can
yield the lock so another queued task can take control of the computer.

Usage:
    async with screen_lock.acquire("task-id"):
        # perform mouse/keyboard actions
        ...
    # lock released — another task can now control the screen

Design:
    - Single global asyncio.Lock wraps all screen interactions
    - Tasks yield the lock during LLM waits (the only time we can switch)
    - Lock holder is tracked for dashboard visibility
    - Timeout prevents deadlocks if a task crashes while holding the lock
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_ACQUIRE_TIMEOUT = float("inf")   # wait forever by default (queue semantics)
_YIELD_GRACE_SECONDS = 0.1        # brief pause after release so waiter can start


class ScreenLock:
    """
    Asyncio-based mutual exclusion for the physical screen.

    Attributes:
        holder_id   : task_id currently holding the lock (or None)
        holder_goal : short description of what the holder is doing
        acquired_at : epoch seconds when lock was last acquired
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.holder_id: str | None = None
        self.holder_goal: str = ""
        self.acquired_at: float = 0.0

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

        Example:
            async with screen_lock.acquire("abc123", "clicking Submit button"):
                computer.mouse.click(x, y)
        """
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Task {task_id} could not acquire the screen lock within {timeout}s"
            )

        self.holder_id = task_id
        self.holder_goal = goal
        self.acquired_at = time.time()
        logger.debug(f"🔒 Screen lock acquired by task {task_id}: {goal}")

        try:
            yield
        finally:
            self.holder_id = None
            self.holder_goal = ""
            self._lock.release()
            logger.debug(f"🔓 Screen lock released by task {task_id}")
            # Brief yield so the next waiter can start before we continue
            await asyncio.sleep(_YIELD_GRACE_SECONDS)

    def is_locked(self) -> bool:
        return self._lock.locked()

    def get_status(self) -> dict:
        return {
            "locked": self.is_locked(),
            "holder_id": self.holder_id,
            "holder_goal": self.holder_goal,
            "held_for_seconds": int(time.time() - self.acquired_at) if self.holder_id else 0,
        }


# ── Global singleton ──────────────────────────────────────────────────────────
# Import this in any module that needs screen access:
#   from autobot.agent.resource_manager import screen_lock

screen_lock = ScreenLock()
