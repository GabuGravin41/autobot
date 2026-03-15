"""
TaskScheduler — Multi-task queue for Autobot.

Manages a queue of agent tasks with:
  - Priority ordering (higher int = runs first)
  - Scheduled start times (run_at: epoch timestamp)
  - Stop conditions (step budget, time limit, metric threshold)
  - Background execution with per-task log capture
  - Cancel / pause / resume
  - Chat-based task injection while another task is running

Architecture:
  - Only ONE task controls the screen (mouse/keyboard) at a time.
  - Parallelism is achieved by time-slicing: task yields the ScreenLock
    during LLM waits so the scheduler can start another task.
  - The scheduler loop ticks every 2s; it checks for:
      1. Waiting tasks whose run_at has passed → move to queued
      2. No active task → start the highest-priority queued task
      3. Active tasks that have finished → mark done/failed

Usage (API layer):
    task_id = await scheduler.add_task("Search for papers on LLMs")
    status  = scheduler.get_task(task_id)
    await scheduler.cancel_task(task_id)
    all_tasks = scheduler.get_all_tasks()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Where to persist the queue
_QUEUE_FILE = Path(__file__).resolve().parent.parent.parent / "runs" / "queue.json"

# Fields serialised to disk (excludes runtime-only fields like logs, current_step)
_PERSIST_FIELDS = {
    "id", "goal", "status", "priority", "run_at",
    "created_at", "started_at", "finished_at", "result", "error",
}

_SCHEDULER_TICK = 2.0          # seconds between scheduler loop ticks
_MAX_LOG_LINES = 500           # per-task log ring buffer size


# ── Task model ────────────────────────────────────────────────────────────────

class TaskStatus(str):
    QUEUED    = "queued"
    SCHEDULED = "scheduled"   # has a future run_at
    STARTING  = "starting"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class ScheduledTask(BaseModel):
    id: str
    goal: str
    status: str = TaskStatus.QUEUED
    priority: int = 1                       # higher = runs first
    run_at: Optional[float] = None          # epoch timestamp; None = ASAP
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None

    # Runtime tracking (not persisted across restarts)
    current_step: int = 0
    max_steps: Optional[int] = None
    eval_signal: str = "continue"
    metrics: Dict[str, float] = Field(default_factory=dict)
    stop_progress: str = ""
    elapsed_seconds: int = 0
    logs: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def summary(self) -> dict:
        """Serialisable summary for API responses."""
        return {
            "id": self.id,
            "goal": self.goal[:120],
            "status": self.status,
            "priority": self.priority,
            "run_at": datetime.fromtimestamp(self.run_at).isoformat() if self.run_at else None,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "started_at": datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            "finished_at": datetime.fromtimestamp(self.finished_at).isoformat() if self.finished_at else None,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "eval_signal": self.eval_signal,
            "metrics": self.metrics,
            "stop_progress": self.stop_progress,
            "elapsed_seconds": self.elapsed_seconds,
            "result": self.result,
            "error": self.error,
        }


# ── Scheduler ─────────────────────────────────────────────────────────────────

class TaskScheduler:
    """
    Global multi-task queue.

    One active task runs at a time (screen is exclusive).
    Other tasks wait in the queue, ordered by priority then created_at.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, ScheduledTask] = {}
        self._runners: Dict[str, Any] = {}          # task_id → AgentRunner
        self._asyncio_tasks: Dict[str, asyncio.Task] = {}  # task_id → asyncio.Task
        self._lock = asyncio.Lock()
        self._loop_task: asyncio.Task | None = None
        self._started = False

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_queue(self) -> None:
        """Write current tasks to disk. Called after every state change."""
        try:
            _QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = []
            for task in self._tasks.values():
                row = {k: v for k, v in task.model_dump().items() if k in _PERSIST_FIELDS}
                data.append(row)
            _QUEUE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save queue: {e}")

    def _load_queue(self) -> None:
        """
        Restore tasks from disk on startup.

        - QUEUED / SCHEDULED tasks are restored as-is so they will be started.
        - RUNNING tasks are demoted back to QUEUED (the runner died with the process).
        - DONE / FAILED / CANCELLED tasks are restored for history only.
        """
        if not _QUEUE_FILE.exists():
            return
        try:
            data = json.loads(_QUEUE_FILE.read_text(encoding="utf-8"))
            for row in data:
                if row.get("status") == TaskStatus.RUNNING:
                    row["status"] = TaskStatus.QUEUED   # re-queue interrupted tasks
                    row["started_at"] = None
                task = ScheduledTask(**row)
                self._tasks[task.id] = task
            pending = sum(1 for t in self._tasks.values() if t.status in (TaskStatus.QUEUED, TaskStatus.SCHEDULED))
            logger.info(f"Queue restored from disk: {len(self._tasks)} tasks ({pending} pending).")
        except Exception as e:
            logger.warning(f"Failed to load queue: {e}")

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler loop. Safe to call multiple times."""
        if self._started:
            return
        self._load_queue()
        self._started = True
        self._loop_task = asyncio.create_task(self._scheduler_loop())
        logger.info("TaskScheduler started.")

    async def stop(self) -> None:
        """Gracefully stop the scheduler loop."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._started = False
        logger.info("TaskScheduler stopped.")

    # ── Public API ─────────────────────────────────────────────────────────

    async def add_task(
        self,
        goal: str,
        priority: int = 1,
        run_at: float | None = None,        # epoch timestamp or None for ASAP
    ) -> str:
        """
        Add a task to the queue.

        Returns the task_id so the caller can poll status or cancel.
        """
        task_id = str(uuid.uuid4())[:8]
        status = TaskStatus.SCHEDULED if run_at and run_at > time.time() else TaskStatus.QUEUED
        task = ScheduledTask(
            id=task_id,
            goal=goal,
            status=status,
            priority=priority,
            run_at=run_at,
        )
        async with self._lock:
            self._tasks[task_id] = task
            self._save_queue()

        _when = f"at {datetime.fromtimestamp(run_at).isoformat()}" if run_at else "immediately"
        logger.info(f"Task queued [{task_id}] ({_when}, priority={priority}): {goal[:80]}")
        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        If it's queued/scheduled → mark cancelled.
        If it's running → stop the runner + asyncio task.
        Returns True if the task was found.
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return True  # already terminal

            task.status = TaskStatus.CANCELLED
            task.finished_at = time.time()

            # Stop the runner
            runner = self._runners.get(task_id)
            if runner:
                try:
                    runner.cancel()
                except Exception:
                    pass

            # Cancel the asyncio task
            at = self._asyncio_tasks.get(task_id)
            if at and not at.done():
                at.cancel()

            self._save_queue()

        logger.info(f"Task cancelled: {task_id}")
        return True

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[dict]:
        return [t.summary() for t in self._tasks.values()]

    def get_active_task(self) -> Optional[ScheduledTask]:
        return next(
            (t for t in self._tasks.values() if t.status == TaskStatus.RUNNING),
            None,
        )

    def get_logs(self, task_id: str, since_line: int = 0) -> List[str]:
        task = self._tasks.get(task_id)
        if not task:
            return []
        return task.logs[since_line:]

    # ── Internal loop ──────────────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}", exc_info=True)
            await asyncio.sleep(_SCHEDULER_TICK)

    async def _tick(self) -> None:
        now = time.time()
        async with self._lock:
            # 1. Promote scheduled tasks whose run_at has arrived
            for task in self._tasks.values():
                if task.status == TaskStatus.SCHEDULED:
                    if task.run_at is not None and now >= task.run_at:
                        task.status = TaskStatus.QUEUED
                        logger.info(f"Task {task.id} promoted from scheduled → queued")

            # 2. Clean up asyncio tasks that finished
            for tid, at in list(self._asyncio_tasks.items()):
                if at.done():
                    self._asyncio_tasks.pop(tid, None)

            # 3. Sync runner status into task model
            for tid, runner in list(self._runners.items()):
                task = self._tasks.get(tid)
                if task and task.status == TaskStatus.RUNNING:
                    try:
                        rs = runner.get_status()
                        task.current_step = rs.get("current_step", task.current_step)
                        task.max_steps = rs.get("max_steps", task.max_steps)
                        task.eval_signal = rs.get("eval_signal", task.eval_signal)
                        task.metrics = rs.get("metrics", task.metrics)
                        task.stop_progress = rs.get("stop_progress", task.stop_progress)
                        task.elapsed_seconds = rs.get("elapsed_seconds", task.elapsed_seconds)
                    except Exception:
                        pass

            # 4. If no task is running, start the next queued one
            active = next(
                (t for t in self._tasks.values() if t.status in (TaskStatus.RUNNING, TaskStatus.STARTING)),
                None,
            )
            if active is None:
                queued = [
                    t for t in self._tasks.values()
                    if t.status == TaskStatus.QUEUED
                ]
                if queued:
                    # Sort by priority DESC, then created_at ASC
                    queued.sort(key=lambda t: (-t.priority, t.created_at))
                    next_task = queued[0]
                    next_task.status = TaskStatus.STARTING
                    # Start outside the lock to avoid deadlocks
                    asyncio.create_task(self._start_task(next_task.id))

    async def _start_task(self, task_id: str) -> None:
        """Launch an AgentRunner for this task in an asyncio task."""
        task = self._tasks.get(task_id)
        if not task:
            return

        from autobot.agent.runner import AgentRunner  # late import avoids circular

        def _log(msg: str) -> None:
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(task.logs) > _MAX_LOG_LINES:
                task.logs = task.logs[-_MAX_LOG_LINES:]

        runner = AgentRunner.from_env(log_callback=_log, task_id=task_id)
        async with self._lock:
            self._runners[task_id] = runner
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

        logger.info(f"Task {task_id} starting: {task.goal[:80]}")

        at = asyncio.create_task(self._run_task(task_id, runner))
        async with self._lock:
            self._asyncio_tasks[task_id] = at

    async def _run_task(self, task_id: str, runner: Any) -> None:
        task = self._tasks.get(task_id)
        if not task:
            return
        try:
            result = await runner.run(task.goal)
            async with self._lock:
                if task.status not in (TaskStatus.CANCELLED,):
                    task.status = TaskStatus.DONE
                    task.result = result
                    task.finished_at = time.time()
                    self._save_queue()
            logger.info(f"Task {task_id} completed.")
        except asyncio.CancelledError:
            async with self._lock:
                if task.status not in (TaskStatus.CANCELLED,):
                    task.status = TaskStatus.CANCELLED
                    task.finished_at = time.time()
                    self._save_queue()
            logger.info(f"Task {task_id} was cancelled.")
        except Exception as e:
            async with self._lock:
                if task.status not in (TaskStatus.CANCELLED,):
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.finished_at = time.time()
                    self._save_queue()
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        finally:
            async with self._lock:
                self._runners.pop(task_id, None)


# ── Global singleton ───────────────────────────────────────────────────────────
scheduler = TaskScheduler()
