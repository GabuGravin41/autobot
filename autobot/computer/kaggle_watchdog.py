"""
Kaggle Job Watchdog — persistent async-job tracking for Kaggle kernels.

Why this exists (empirical, not speculative): Dalton ran three real
competitions through Antigravity against his real Kaggle account
(biohub_cell_tracking, ieee_traffic_flow, s6e9_ev_prediction — see
competitions/*/THINKING_AND_DECISIONS.md) specifically to find out where an
LLM-driven agent breaks on real infrastructure. The single most costly,
repeated failure documented there — not a hypothetical, an actual incident
at 23:34 during the S6E9 run — was this:

    A TabPFN kernel was pushed. A status check returned RUNNING. The agent
    told the user it was running and moved on to the next task. 19 seconds
    later the kernel crashed (TabPFNLicenseError). The agent had no way to
    find out — it doesn't run as a daemon, so once its turn ends it goes
    fully dormant until the user speaks again — and the failure sat
    undetected until the user manually asked "check if it's really still
    running."

Root cause, in the user's own diagnosis: an LLM agent is a turn-based
reactive process (input -> reason -> tool calls -> message -> HALT), not a
continuous loop. Kaggle does not push webhooks on kernel failure. ~80% of
cloud job failures happen in the first 60 seconds (missing deps, license
checks, OOM on data load, path errors) — exactly the window an agent is
most likely to have already stopped watching.

This module is the "deterministic software scaffolding" the lessons doc
calls for, in the smallest form that actually closes the gap:

1. A persistent, on-disk ledger (survives process restarts and dead
   conversations — the point is that the TRUTH doesn't live in the LLM's
   context window).
2. `verify_liveness()` — a short, bounded grace-period check (t+30s,
   t+60s by default) that must be run before a kernel is ever reported as
   successfully "running." This is the documented "60-Second Liveness
   Verification Rule," applied as code instead of as a hope that the model
   remembers to do it.
3. `poll_pending()` — a cheap, non-blocking sweep across every job the
   ledger knows about, for something else (a background task, a CLI
   command, a dashboard) to call on its own schedule, so status changes
   don't depend on the agent's conversation still being open.

What this module deliberately does NOT do: run its own thread, own clock,
or own retry loop. It is pure state + one bounded blocking check. Something
long-lived has to call `poll_pending()` repeatedly for the "decoupled
daemon" picture to be real — see autobot/web/app.py's lifespan task, which
is the actual daemon (the FastAPI server is the one part of Autobot that
runs as a persistent process rather than a one-shot CLI turn).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Terminal states a job doesn't need re-polling from once reached.
_TERMINAL_STATUSES = {"complete", "error", "cancelled"}

DEFAULT_LEDGER_PATH = Path(os.path.expanduser("~/.autobot/kaggle_jobs.json"))

# Kaggle Cloud's own hard per-account hardware caps, as documented (and hit
# in practice) in s6e9_ev_prediction/THINKING_AND_DECISIONS.md section 5.C
# ("The Compute Resource Semaphore — The Kaggle Cloud Operating System"):
# max 2 concurrent GPU kernels (T4/P100), max 4 concurrent CPU kernels,
# across the WHOLE account — not per project. Dispatching a 3rd GPU kernel
# doesn't run faster in a queue Kaggle manages gracefully for you; it's
# either rejected outright or silently queued behind kernels Autobot has
# already lost track of wanting to run first. Enforcing this locally, before
# ever shelling out to push_kernel, is what actually makes "run multiple
# Kaggle-competition projects at once" safe rather than a way to
# accidentally oversubscribe a shared account-wide quota.
KAGGLE_GPU_SLOT_LIMIT = 2
KAGGLE_CPU_SLOT_LIMIT = 4


@dataclass
class JobRecord:
    kernel: str
    competition: str | None = None
    path: str | None = None
    status: str = "dispatched"          # dispatched|running|complete|error|cancelled|unknown
    hardware: str | None = None         # "gpu" | "cpu" | None (unknown / not tracked)
    dispatched_at: float = field(default_factory=time.time)
    last_checked_at: float | None = None
    liveness_verified: bool = False
    error_log: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, status: str) -> None:
        self.status = status
        self.last_checked_at = time.time()
        self.history.append({"at": self.last_checked_at, "status": status})
        # Keep the ledger file from growing unbounded across a long-lived job.
        self.history = self.history[-50:]


class KaggleJobLedger:
    """
    JSON-backed store of every kernel job Autobot has dispatched, keyed by
    kernel slug. Load/save is intentionally simple (whole-file read/write
    with an atomic rename) — job counts are small (tens, not thousands),
    so there's no need for a real database here.
    """

    def __init__(self, ledger_path: str | Path = DEFAULT_LEDGER_PATH):
        self.path = Path(ledger_path)
        self._jobs: dict[str, JobRecord] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._jobs = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._jobs = {k: JobRecord(**v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, OSError) as e:
            # A corrupt or half-written ledger must never crash a run —
            # start fresh and log loudly, don't silently swallow.
            logger.error(f"Kaggle job ledger at {self.path} unreadable ({e}); starting fresh.")
            self._jobs = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        payload = {k: asdict(v) for k, v in self._jobs.items()}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic on both POSIX and Windows

    def register(
        self,
        kernel: str,
        competition: str | None = None,
        path: str | None = None,
        hardware: str | None = None,
    ) -> JobRecord:
        job = JobRecord(kernel=kernel, competition=competition, path=path, hardware=hardware)
        self._jobs[kernel] = job
        self.save()
        return job

    def active_hardware_count(self, hardware: str) -> int:
        """How many non-terminal jobs are currently occupying a slot of this
        hardware type (`\"gpu\"` or `\"cpu\"`) — the live number to check
        against KAGGLE_GPU_SLOT_LIMIT/KAGGLE_CPU_SLOT_LIMIT before dispatching
        another. Jobs with no recorded hardware (registered before Round 8,
        or by a caller that didn't pass it) don't count either way — they
        were never tracked for capacity, so counting them would either
        under- or over-count depending on what they actually were; the
        honest answer is that this ledger doesn't know."""
        return sum(1 for j in self.pending() if j.hardware == hardware)

    def get(self, kernel: str) -> JobRecord | None:
        return self._jobs.get(kernel)

    def all(self) -> list[JobRecord]:
        return list(self._jobs.values())

    def pending(self) -> list[JobRecord]:
        return [j for j in self._jobs.values() if j.status not in _TERMINAL_STATUSES]

    def update(self, kernel: str, status: str, error_log: str | None = None) -> JobRecord:
        job = self._jobs.get(kernel) or self.register(kernel)
        job.record(status)
        if error_log is not None:
            job.error_log = error_log
        self.save()
        return job


def _normalize_status(raw: Any) -> str:
    """KaggleApi.kernels_status() return shape has drifted before (see
    kaggle_tool.py's list_competitions() docstring) — normalize defensively
    rather than assuming a fixed type."""
    s = str(raw).lower()
    if "error" in s or "fail" in s:
        return "error"
    if "complete" in s or "ok" in s or "success" in s:
        return "complete"
    if "running" in s:
        return "running"
    if "queue" in s:
        return "queued"
    if "cancel" in s:
        return "cancelled"
    return "unknown"


def verify_liveness(
    api: Any,
    ledger: KaggleJobLedger,
    kernel: str,
    grace_checks: tuple[int, ...] = (30, 60),
    sleep_fn: Callable[[float], None] = time.sleep,
    fetch_error_log: Callable[[Any, str], str | None] | None = None,
) -> dict[str, Any]:
    """
    The 60-Second Liveness Verification Rule, as code.

    Blocks for up to `grace_checks[-1]` seconds total (default 60s) —
    this is a short, bounded wait, not the long poll loop the lessons doc
    warns against saturating context with. It exists specifically to catch
    the ~80%-of-failures-in-the-first-60-seconds window before anything
    downstream treats the job as safely launched.

    `grace_checks` are deltas from the call, e.g. (30, 60) checks once at
    t+30s and again at t+60s (30s apart), matching the documented pattern.
    Pass a fake `sleep_fn` in tests to avoid real waiting.

    Returns {"survived_init": bool, "status": str, "checked_at": [..]}.
    Does NOT raise on ERROR — a failed liveness check is information the
    caller (kaggle_tool.py's push_kernel) surfaces to whoever's driving,
    since deciding what to do about a dead-on-arrival kernel is a strategy
    call, not this module's job.
    """
    job = ledger.get(kernel) or ledger.register(kernel)
    checked_at: list[int] = []
    prev_delay = 0
    status = "unknown"

    for delay in grace_checks:
        sleep_fn(delay - prev_delay)
        prev_delay = delay
        raw_status = api.kernels_status(kernel)
        status = _normalize_status(raw_status)
        checked_at.append(delay)

        if status == "error":
            error_log = None
            if fetch_error_log is not None:
                try:
                    error_log = fetch_error_log(api, kernel)
                except Exception as e:
                    error_log = f"<could not fetch error log: {e}>"
            ledger.update(kernel, "error", error_log=error_log)
            return {
                "survived_init": False,
                "status": "error",
                "checked_at": checked_at,
                "error_log": error_log,
            }

        if status in ("running", "complete"):
            job.liveness_verified = True
            ledger.update(kernel, status)
            return {"survived_init": True, "status": status, "checked_at": checked_at}

    # Ran through every grace check without a clear RUNNING/COMPLETE/ERROR —
    # report honestly rather than guessing either way.
    ledger.update(kernel, status)
    return {"survived_init": status != "error", "status": status, "checked_at": checked_at}


_HARDWARE_LIMITS = {"gpu": KAGGLE_GPU_SLOT_LIMIT, "cpu": KAGGLE_CPU_SLOT_LIMIT}


def check_capacity(ledger: KaggleJobLedger, hardware: str | None) -> tuple[bool, str]:
    """
    Whether dispatching one more `hardware`-tier kernel would exceed
    Kaggle's account-wide slot limit, per the ledger's current view of
    what's active. Returns (has_capacity, message) — the message is
    informational either way (e.g. "1/2 GPU slots in use"), not just an
    error string, so a caller can log it on the happy path too.

    `hardware=None` (unknown/not tracked) always reports capacity — this
    function enforces a KNOWN constraint, it doesn't invent one for jobs
    this ledger was never told the hardware tier of.
    """
    if hardware is None or hardware not in _HARDWARE_LIMITS:
        return True, "hardware tier not tracked — capacity not enforced"
    limit = _HARDWARE_LIMITS[hardware]
    active = ledger.active_hardware_count(hardware)
    if active >= limit:
        return False, f"{active}/{limit} {hardware.upper()} slots in use — at capacity"
    return True, f"{active}/{limit} {hardware.upper()} slots in use"


def poll_pending(api: Any, ledger: KaggleJobLedger) -> list[JobRecord]:
    """
    Cheap, non-blocking sweep: check current status of every non-terminal
    job in the ledger, update it, and return only the ones whose status
    just changed. Meant to be called on its own schedule by something
    long-lived (the FastAPI server's background task, or `autobot --jobs`
    on demand) — not by the agent's own conversational turn, which is
    exactly the mechanism the lessons doc identified as unreliable for
    this.
    """
    changed: list[JobRecord] = []
    for job in ledger.pending():
        try:
            raw_status = api.kernels_status(job.kernel)
        except Exception as e:
            logger.warning(f"poll_pending: status check failed for {job.kernel}: {e}")
            continue
        new_status = _normalize_status(raw_status)
        if new_status != job.status:
            ledger.update(job.kernel, new_status)
            changed.append(job)
    return changed
