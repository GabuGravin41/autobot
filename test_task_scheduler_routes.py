"""Behavioral test for the task queue API (agent/scheduler.py wired into web/app.py).

Root cause fixed here: apiService.ts's QueuedTask/ScheduleStatus/ScreenLockStatus
interfaces (and TaskQueuePanel.tsx built on top of them) have called
/api/tasks*, /api/schedule/status, and /api/screen-lock since before this
session started. None of those routes existed in app.py — confirmed by
diffing every apiFetch() call against every registered route. Meanwhile
agent/scheduler.py's TaskScheduler and agent/resource_manager.py's
ScreenLock already implement almost exactly this contract field-for-field.
Both halves were complete; only the connecting routes were missing.

This drives the REAL FastAPI app over its ASGI interface via httpx (no
real TCP port needed, unlike the manual uvicorn session used to first
verify this live) - same production code path, fully repeatable.

Also verifies the concurrency safety fix: real concurrent execution is
capped at 1 regardless of AUTOBOT_MAX_CONCURRENT_TASKS, because nothing in
AgentLoop's action dispatch actually acquires screen_lock yet - letting N
tasks run "concurrently" without that would mean N AgentRunners fighting
over the same physical mouse/keyboard with zero coordination.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


async def main():
    import httpx
    from autobot.web.app import app
    from autobot.agent.scheduler import scheduler

    # Route registration alone doesn't start the tick loop (that's the
    # lifespan hook, which httpx's ASGITransport doesn't trigger) - start it
    # explicitly so add_task() results actually get picked up, same as a
    # real running server.
    scheduler.start()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # ---- queue endpoint responds correctly ----
        # Not asserting an EMPTY queue: agent/scheduler.py deliberately
        # persists tasks to runs/queue.json across restarts (a real feature,
        # not a test artifact) — a real user's queue could legitimately have
        # history in it, and the test shouldn't require a pristine machine.
        r = await client.get("/api/tasks")
        check("GET /api/tasks returns 200 with a tasks list",
              r.status_code == 200 and isinstance(r.json().get("tasks"), list))
        baseline_count = len(r.json()["tasks"])

        # ---- schedule status shape matches ScheduleStatus exactly ----
        r = await client.get("/api/schedule/status")
        status = r.json()
        check("GET /api/schedule/status has the exact expected shape",
              set(status.keys()) == {"max_concurrent", "slots_used", "slots_free",
                                      "running", "queued", "paused", "screen_lock"},
              str(status.keys()))
        check("real concurrency is capped at 1 (screen_lock not wired into AgentLoop yet)",
              status["max_concurrent"] == 1, str(status))

        # ---- screen-lock shape matches ScreenLockStatus exactly ----
        r = await client.get("/api/screen-lock")
        lock = r.json()
        check("GET /api/screen-lock has the exact expected shape",
              set(lock.keys()) == {"locked", "holder_id", "holder_goal", "held_for_seconds",
                                    "last_released_by", "waiting_tasks"},
              str(lock.keys()))

        # ---- add a far-future scheduled task (won't race the tick loop) ----
        future = time.time() + 3600
        r = await client.post("/api/tasks", json={"goal": "test scheduled task", "priority": 1, "run_at": future})
        check("POST /api/tasks returns 200 with a task_id", r.status_code == 200 and "task_id" in r.json())
        task_id = r.json()["task_id"]

        r = await client.get(f"/api/tasks/{task_id}")
        detail = r.json()
        check("scheduled task's fields match QueuedTask exactly",
              set(detail.keys()) == {"id", "goal", "status", "priority", "run_at", "created_at",
                                      "started_at", "finished_at", "current_step", "max_steps",
                                      "eval_signal", "metrics", "stop_progress", "elapsed_seconds",
                                      "result", "error"},
              str(detail.keys()))
        check("a future run_at task is 'scheduled', not started early",
              detail["status"] == "scheduled", detail["status"])

        # ---- appears in the list, and the count grew by exactly one ----
        r = await client.get("/api/tasks")
        tasks_now = r.json()["tasks"]
        check("newly added task appears in GET /api/tasks",
              any(t["id"] == task_id for t in tasks_now))
        check("task count grew by exactly one",
              len(tasks_now) == baseline_count + 1, f"{len(tasks_now)} vs baseline {baseline_count}")

        # ---- priority change persists ----
        r = await client.patch(f"/api/tasks/{task_id}/priority", params={"priority": 5})
        check("PATCH priority succeeds", r.status_code == 200)
        r = await client.get(f"/api/tasks/{task_id}")
        check("priority change actually persisted", r.json()["priority"] == 5, str(r.json()["priority"]))

        # ---- empty-goal rejected ----
        r = await client.post("/api/tasks", json={"goal": "   "})
        check("empty goal is rejected with 400", r.status_code == 400)

        # ---- cancel, then confirm terminal state ----
        r = await client.delete(f"/api/tasks/{task_id}")
        check("DELETE cancels the task", r.status_code == 200 and r.json()["status"] == "cancelled")
        r = await client.get(f"/api/tasks/{task_id}")
        check("cancelled task shows status=cancelled", r.json()["status"] == "cancelled")

        # ---- operating on a nonexistent task fails cleanly, not with a 500 ----
        r = await client.get("/api/tasks/does-not-exist")
        check("GET on unknown task -> 404, not 500", r.status_code == 404)
        r = await client.delete("/api/tasks/does-not-exist")
        check("DELETE on unknown task -> 404, not 500", r.status_code == 404)
        r = await client.post("/api/tasks/does-not-exist/pause")
        check("pause on unknown task -> 400 (not found/not pausable), not 500", r.status_code == 400)
        r = await client.post("/api/tasks/does-not-exist/resume")
        check("resume on unknown task -> 400, not 500", r.status_code == 400)

        # ---- logs endpoint shape ----
        r = await client.get(f"/api/tasks/{task_id}/logs", params={"since": 0})
        check("GET logs has the {lines, total} shape the frontend expects",
              set(r.json().keys()) == {"lines", "total"}, str(r.json()))

    await scheduler.stop()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


asyncio.run(main())
