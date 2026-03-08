"""
Scheduler — Manages multiple concurrent or queued agent tasks.
Allows switching context when a task is waiting.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from autobot.agent.runner import AgentRunner

logger = logging.getLogger(__name__)

class ScheduledTask(BaseModel):
    id: str
    goal: str
    status: str = "queued" # queued | running | waiting | done | failed
    created_at: str
    started_at: Optional[str] = None
    priority: int = 1
    wait_until: Optional[float] = None # timestamp

class TaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.runners: Dict[str, AgentRunner] = {}
        self._loop_task = None
        self._lock = asyncio.Lock()

    def start(self):
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._scheduler_loop())
            logger.info("Task scheduler started.")

    async def add_task(self, goal: str, priority: int = 1) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = ScheduledTask(
            id=task_id,
            goal=goal,
            status="queued",
            created_at=datetime.now().isoformat(),
            priority=priority
        )
        async with self._lock:
            self.tasks[task_id] = task
        logger.info(f"Task queued: {task_id} - {goal}")
        return task_id

    async def _scheduler_loop(self):
        while True:
            try:
                await self._check_and_run_tasks()
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            await asyncio.sleep(5)

    async def _check_and_run_tasks(self):
        async with self._lock:
            # 1. Check for tasks that are "waiting" and their time is up
            now = asyncio.get_event_loop().time()
            for tid, task in self.tasks.items():
                if task.status == "waiting" and task.wait_until and now >= task.wait_until:
                    task.status = "queued"
                    task.wait_until = None
                    logger.info(f"Task {tid} resumed from wait.")

            # 2. If no task is running, pick the highest priority queued task
            running_task = next((t for t in self.tasks.values() if t.status == "running"), None)
            
            if not running_task:
                queued_tasks = [t for t in self.tasks.values() if t.status == "queued"]
                if queued_tasks:
                    queued_tasks.sort(key=lambda x: x.priority, reverse=True)
                    next_task = queued_tasks[0]
                    await self._start_task(next_task)

    async def _start_task(self, task: ScheduledTask):
        task.status = "running"
        task.started_at = datetime.now().isoformat()
        
        runner = AgentRunner.from_env()
        self.runners[task.id] = runner
        
        # Run in background
        asyncio.create_task(self._run_runner(task, runner))
        logger.info(f"Task {task.id} started execution.")

    async def _run_runner(self, task: ScheduledTask, runner: AgentRunner):
        try:
            # Note: runner.run is likely blocking/async, need to ensure it yields
            result = await runner.run(task.goal)
            async with self._lock:
                task.status = "done"
                logger.info(f"Task {task.id} completed.")
        except Exception as e:
            async with self._lock:
                task.status = "failed"
                logger.error(f"Task {task.id} failed: {e}")

    def get_tasks(self) -> List[ScheduledTask]:
        return list(self.tasks.values())

    async def cancel_task(self, task_id: str):
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = "cancelled"
                if task_id in self.runners:
                    self.runners[task_id].cancel()
                logger.info(f"Task {task_id} cancelled.")

# Global instance
scheduler = TaskScheduler()
