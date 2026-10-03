"""
The butler daemon: the part that owns TIME, so no model has to.

One process, started with `autobot butler run`, that loops forever:

  1. heartbeat (so `autobot butler status` / the dashboard can tell it's alive)
  2. apply your inbox decisions (approvals, answers, rejections)
  3. claim due tasks and run one playbook step for each, in worker threads,
     bounded by AUTOBOT_BUTLER_MAX_CONCURRENT (default 2 — your Claude and
     Antigravity plans have usage limits; more parallelism mostly buys
     rate-limit cooldowns)
  4. sleep a few seconds

Guarantees:
  - Single instance: a lock file stops a second daemon from starting.
  - Crash-safe: every step's outcome is in the database before the next
    begins; tasks caught mid-step by a crash are resumed on the next start.
  - A step that raises never kills the daemon; the task backs off and the
    error is in its history.
  - Stopping (`autobot butler stop`, Ctrl+C) kills the worker processes it
    started instead of leaving them running unattended.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from autobot.butler.playbook import Playbook
from autobot.butler.store import QUEUED, WORKING, ButlerStore, Task

logger = logging.getLogger(__name__)

STEP_ERROR_BACKOFF_S = 300.0


class AlreadyRunning(RuntimeError):
    pass


class ButlerDaemon:
    def __init__(
        self,
        store: ButlerStore | None = None,
        playbook: Playbook | None = None,
        llm: Any | None = "auto",
        max_concurrent: int | None = None,
        tick_seconds: float = 5.0,
        log=print,
    ) -> None:
        self.store = store or ButlerStore()
        if llm == "auto":
            try:
                from autobot.llm import get_manager_llm
                llm = get_manager_llm()
            except Exception as e:  # never let manager-LLM config stop the daemon
                logger.warning(f"manager LLM unavailable: {e}")
                llm = None
        self.llm = llm
        self.playbook = playbook or Playbook(self.store, llm=llm)
        self.max_concurrent = max_concurrent or int(os.getenv("AUTOBOT_BUTLER_MAX_CONCURRENT", "2"))
        self.tick_seconds = tick_seconds
        self.log = log
        self._futures: dict[int, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._stop = threading.Event()
        self._lock = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _stop_file(self):
        from autobot.paths import autobot_home
        return autobot_home() / "butler.stop"

    def acquire_single_instance(self) -> None:
        from autobot.paths import autobot_home
        from autobot.util.filelock import FileLock, LockTimeout
        lock = FileLock(autobot_home() / "butler.lock", timeout=1, reentrant=False)
        try:
            lock.acquire()
        except LockTimeout:
            raise AlreadyRunning("Another butler daemon is already running on this machine.") from None
        self._lock = lock

    def run_forever(self) -> None:
        self.acquire_single_instance()
        try:
            self._stop_file().unlink()
        except FileNotFoundError:
            pass
        recovered = self.store.recover_orphans()
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent, thread_name_prefix="butler")
        self.store.set_meta("daemon", {"pid": os.getpid(), "started_at": time.time(),
                                       "llm": self.llm.describe() if self.llm else None,
                                       "max_concurrent": self.max_concurrent})
        self.log(f"Autobot butler running (pid {os.getpid()}). Manager model: "
                 f"{self.llm.describe() if self.llm else 'none — playbooks only, questions go to your inbox'}.")
        if recovered:
            self.log(f"Resuming {len(recovered)} task(s) interrupted by the last shutdown: {recovered}")
        try:
            while not self._stop.is_set():
                if self._stop_file().exists():
                    self.log("Stop requested.")
                    break
                self.tick()
                self._stop.wait(self.tick_seconds)
        except KeyboardInterrupt:
            self.log("Interrupted.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        from autobot.integrations.cli_exec import kill_all_active
        self._stop.set()
        killed = kill_all_active()
        if killed:
            self.log(f"Stopped {killed} running worker process(es); their tasks resume on next start.")
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self.store.recover_orphans()
        self.store.set_meta("daemon", {"pid": None, "stopped_at": time.time()})
        if self._lock:
            self._lock.release()
            self._lock = None
        try:
            self._stop_file().unlink()
        except FileNotFoundError:
            pass

    def stop(self) -> None:
        self._stop.set()

    # ── one tick ────────────────────────────────────────────────────────────

    def tick(self) -> int:
        """One scheduling pass. Returns how many steps were started (or run inline)."""
        self.store.set_meta("heartbeat", time.time())
        for approval in self.store.unhandled_decisions():
            try:
                self.playbook.on_decision(approval)
            except Exception as e:
                self.store.log(approval.task_id, "error", f"Handling your decision failed: {e}")
                self.store.mark_handled(approval.id)

        self._futures = {tid: f for tid, f in self._futures.items() if not f.done()}
        free = self.max_concurrent - len(self._futures)
        if free <= 0:
            return 0
        started = 0
        for task in self.store.claim_due(limit=free):
            if self._executor is None:
                self._run_step(task)             # inline mode (tests, run_until_idle)
            else:
                self._futures[task.id] = self._executor.submit(self._run_step, task)
            started += 1
        return started

    def _run_step(self, task: Task) -> None:
        try:
            self.playbook.step(task)
        except Exception as e:
            tb = traceback.format_exc()
            self.store.log(task.id, "error", f"Step crashed: {type(e).__name__}: {e}", {"traceback": tb[-4000:]})
            self.store.update_task(task.id, status=QUEUED, next_wake_at=time.time() + STEP_ERROR_BACKOFF_S)
        finally:
            current = self.store.get_task(task.id)
            if current is not None and current.status == WORKING:
                # A step must always leave a resting status; if it didn't, don't strand the task.
                self.store.log(task.id, "error", "Step ended without choosing a next state; retrying later.")
                self.store.update_task(task.id, status=QUEUED, next_wake_at=time.time() + STEP_ERROR_BACKOFF_S)

    def run_until_idle(self, max_ticks: int = 200) -> int:
        """Synchronous mode: keep ticking until nothing is due. For tests and one-shot runs."""
        total = 0
        for _ in range(max_ticks):
            n = self.tick()
            total += n
            if n == 0 and not self.store.unhandled_decisions():
                break
        return total
