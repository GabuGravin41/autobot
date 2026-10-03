"""
Durable state for the butler: tasks, their event history, approvals and
questions for the user, and resource cooldowns — one SQLite file at
~/.autobot/butler.db (see autobot/paths.py).

Why a database and not the conversation: the Kaggle post-mortems showed an
LLM agent is turn-based, not a daemon. State held "in the conversation"
dies with the turn; state on disk survives restarts, crashes, and the model
forgetting. Every decision the butler makes is recomputed from this file.

Concurrency: every operation opens its own short-lived connection (safe
across the daemon's worker threads and across processes — the web server
and CLI read and approve through the same file). WAL mode lets readers
proceed while the daemon writes. Claiming work uses BEGIN IMMEDIATE so two
daemons (or two threads) can never start the same task twice.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lane          TEXT NOT NULL,
    title         TEXT NOT NULL,
    intent        TEXT NOT NULL,
    project_dir   TEXT,
    worker        TEXT,
    status        TEXT NOT NULL DEFAULT 'queued',
    priority      INTEGER NOT NULL DEFAULT 5,
    criteria      TEXT NOT NULL DEFAULT '[]',
    criteria_source TEXT NOT NULL DEFAULT 'none',
    config        TEXT NOT NULL DEFAULT '{}',
    state         TEXT NOT NULL DEFAULT '{}',
    next_wake_at  REAL NOT NULL DEFAULT 0,
    attempts      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    result        TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id  INTEGER,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,
    message  TEXT NOT NULL,
    data     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, id);
CREATE TABLE IF NOT EXISTS approvals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      INTEGER,
    kind         TEXT NOT NULL,
    summary      TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'pending',
    created_at   REAL NOT NULL,
    decided_at   REAL,
    decided_via  TEXT,
    response     TEXT,
    handled      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
CREATE TABLE IF NOT EXISTS resources (
    name           TEXT PRIMARY KEY,
    cooldown_until REAL NOT NULL DEFAULT 0,
    reason         TEXT,
    updated_at     REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Task statuses
QUEUED = "queued"        # ready to run as soon as it's due
WORKING = "working"      # a step is executing right now (worker or checks)
WAITING = "waiting"      # sleeping until next_wake_at (e.g. a Kaggle kernel is running)
NEEDS_YOU = "needs_you"  # blocked on an approval or a question in the inbox
PAUSED = "paused"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

ACTIVE_STATUSES = (QUEUED, WORKING, WAITING, NEEDS_YOU, PAUSED)
TERMINAL_STATUSES = (DONE, FAILED, CANCELLED)


@dataclass
class Task:
    id: int
    lane: str
    title: str
    intent: str
    project_dir: str | None
    worker: str | None
    status: str
    priority: int
    criteria: list[dict]
    criteria_source: str
    config: dict
    state: dict
    next_wake_at: float
    attempts: int
    created_at: float
    updated_at: float
    result: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        return cls(
            id=row["id"], lane=row["lane"], title=row["title"], intent=row["intent"],
            project_dir=row["project_dir"], worker=row["worker"], status=row["status"],
            priority=row["priority"], criteria=json.loads(row["criteria"] or "[]"),
            criteria_source=row["criteria_source"], config=json.loads(row["config"] or "{}"),
            state=json.loads(row["state"] or "{}"), next_wake_at=row["next_wake_at"],
            attempts=row["attempts"], created_at=row["created_at"], updated_at=row["updated_at"],
            result=row["result"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Approval:
    id: int
    task_id: int | None
    kind: str
    summary: str
    payload: dict
    status: str
    created_at: float
    decided_at: float | None
    decided_via: str | None
    response: str | None
    handled: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Approval":
        return cls(
            id=row["id"], task_id=row["task_id"], kind=row["kind"], summary=row["summary"],
            payload=json.loads(row["payload"] or "{}"), status=row["status"],
            created_at=row["created_at"], decided_at=row["decided_at"], decided_via=row["decided_via"],
            response=row["response"], handled=bool(row["handled"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Event:
    id: int
    task_id: int | None
    ts: float
    kind: str
    message: str
    data: dict = field(default_factory=dict)


class ButlerStore:
    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            from autobot.paths import butler_db_path
            path = butler_db_path()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    # ── tasks ───────────────────────────────────────────────────────────────

    def add_task(
        self,
        lane: str,
        title: str,
        intent: str,
        project_dir: str | None = None,
        worker: str | None = None,
        criteria: list[dict] | None = None,
        config: dict | None = None,
        priority: int = 5,
        start_at: float | None = None,
    ) -> Task:
        if not intent or not intent.strip():
            raise ValueError("A task needs an intent: what you want, in your own words.")
        now = time.time()
        criteria = criteria or []
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO tasks (lane, title, intent, project_dir, worker, criteria, criteria_source,"
                " config, priority, next_wake_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (lane, title.strip() or intent.strip()[:80], intent, project_dir, worker,
                 json.dumps(criteria), "user" if criteria else "none", json.dumps(config or {}),
                 int(priority), start_at or now, now, now),
            )
            task_id = cur.lastrowid
        self.log(task_id, "created", f"Task created in lane '{lane}'",
                 {"worker": worker, "project_dir": project_dir, "criteria": criteria})
        return self.get_task(task_id)  # type: ignore[return-value]

    def get_task(self, task_id: int) -> Task | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return Task.from_row(row) if row else None

    def list_tasks(self, statuses: tuple[str, ...] | None = None, limit: int = 200) -> list[Task]:
        q = "SELECT * FROM tasks"
        args: list[Any] = []
        if statuses:
            q += f" WHERE status IN ({','.join('?' * len(statuses))})"
            args += list(statuses)
        q += " ORDER BY CASE WHEN status IN ('done','failed','cancelled') THEN 1 ELSE 0 END, priority, id LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [Task.from_row(r) for r in rows]

    _JSON_FIELDS = ("criteria", "config", "state")
    _ALLOWED_FIELDS = {"lane", "title", "intent", "project_dir", "worker", "status", "priority", "criteria",
                       "criteria_source", "config", "state", "next_wake_at", "attempts", "result"}

    def update_task(self, task_id: int, **fields: Any) -> Task | None:
        bad = set(fields) - self._ALLOWED_FIELDS
        if bad:
            raise ValueError(f"unknown task fields: {bad}")
        if not fields:
            return self.get_task(task_id)
        sets, args = [], []
        for k, v in fields.items():
            sets.append(f"{k}=?")
            args.append(json.dumps(v) if k in self._JSON_FIELDS else v)
        sets.append("updated_at=?")
        args.append(time.time())
        args.append(task_id)
        with self._conn() as c:
            c.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", args)
        return self.get_task(task_id)

    def merge_state(self, task_id: int, **changes: Any) -> dict:
        """Read-modify-write of the task's playbook state, inside one transaction."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT state FROM tasks WHERE id=?", (task_id,)).fetchone()
            state = json.loads(row["state"] or "{}") if row else {}
            state.update(changes)
            c.execute("UPDATE tasks SET state=?, updated_at=? WHERE id=?", (json.dumps(state), time.time(), task_id))
            c.execute("COMMIT")
        return state

    def claim_due(self, now: float | None = None, limit: int = 1,
                  exclude_lanes: tuple[str, ...] = ()) -> list[Task]:
        """Atomically move up to `limit` due tasks to WORKING and return them.
        Highest priority (lowest number) first, then oldest."""
        now = time.time() if now is None else now
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            q = ("SELECT * FROM tasks WHERE status IN (?, ?) AND next_wake_at <= ?")
            args: list[Any] = [QUEUED, WAITING, now]
            if exclude_lanes:
                q += f" AND lane NOT IN ({','.join('?' * len(exclude_lanes))})"
                args += list(exclude_lanes)
            q += " ORDER BY priority, next_wake_at, id LIMIT ?"
            args.append(limit)
            rows = c.execute(q, args).fetchall()
            for r in rows:
                c.execute("UPDATE tasks SET status=?, updated_at=? WHERE id=?", (WORKING, now, r["id"]))
            c.execute("COMMIT")
        return [self.get_task(r["id"]) for r in rows]  # type: ignore[misc]

    def recover_orphans(self) -> list[int]:
        """Tasks left WORKING by a daemon that died mid-step go back to QUEUED."""
        with self._conn() as c:
            rows = c.execute("SELECT id FROM tasks WHERE status=?", (WORKING,)).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                c.execute(f"UPDATE tasks SET status=?, next_wake_at=?, updated_at=? WHERE id IN ({','.join('?' * len(ids))})",
                          [QUEUED, time.time(), time.time(), *ids])
        for i in ids:
            self.log(i, "recovered", "The daemon stopped while this task was mid-step; it will resume.")
        return ids

    # ── events ──────────────────────────────────────────────────────────────

    def log(self, task_id: int | None, kind: str, message: str, data: dict | None = None) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO events (task_id, ts, kind, message, data) VALUES (?,?,?,?,?)",
                      (task_id, time.time(), kind, message, json.dumps(data or {}, default=str)))

    def events(self, task_id: int | None = None, limit: int = 50, since_id: int = 0) -> list[Event]:
        q = "SELECT * FROM events WHERE id > ?"
        args: list[Any] = [since_id]
        if task_id is not None:
            q += " AND task_id = ?"
            args.append(task_id)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [Event(r["id"], r["task_id"], r["ts"], r["kind"], r["message"], json.loads(r["data"] or "{}"))
                for r in reversed(rows)]

    # ── approvals & questions (the inbox) ───────────────────────────────────

    def add_approval(self, task_id: int | None, kind: str, summary: str, payload: dict | None = None) -> Approval:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO approvals (task_id, kind, summary, payload, created_at) VALUES (?,?,?,?,?)",
                (task_id, kind, summary, json.dumps(payload or {}, default=str), time.time()),
            )
            aid = cur.lastrowid
        self.log(task_id, "inbox", f"[{kind}] {summary}", {"approval_id": aid})
        return self.get_approval(aid)  # type: ignore[return-value]

    def get_approval(self, approval_id: int) -> Approval | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
        return Approval.from_row(row) if row else None

    def approvals(self, status: str | None = "pending", task_id: int | None = None, limit: int = 100) -> list[Approval]:
        q, args = "SELECT * FROM approvals WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if task_id is not None:
            q += " AND task_id=?"
            args.append(task_id)
        q += " ORDER BY id LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [Approval.from_row(r) for r in rows]

    def decide(self, approval_id: int, approved: bool, via: str = "cli", response: str | None = None) -> Approval:
        """Record the user's decision. Only a PENDING item can be decided — a
        decision can't be silently overwritten later."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT status FROM approvals WHERE id=?", (approval_id,)).fetchone()
            if row is None:
                c.execute("ROLLBACK")
                raise KeyError(f"No inbox item #{approval_id}")
            if row["status"] != "pending":
                c.execute("ROLLBACK")
                raise ValueError(f"Inbox item #{approval_id} was already {row['status']}")
            c.execute("UPDATE approvals SET status=?, decided_at=?, decided_via=?, response=? WHERE id=?",
                      ("approved" if approved else "rejected", time.time(), via, response, approval_id))
            c.execute("COMMIT")
        a = self.get_approval(approval_id)
        assert a is not None
        self.log(a.task_id, "decision", f"You {'approved' if approved else 'rejected'} [{a.kind}] #{a.id}",
                 {"approval_id": a.id, "response": response, "via": via})
        return a

    def unhandled_decisions(self) -> list[Approval]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM approvals WHERE status IN ('approved','rejected') AND handled=0 ORDER BY id").fetchall()
        return [Approval.from_row(r) for r in rows]

    def mark_handled(self, approval_id: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE approvals SET handled=1 WHERE id=?", (approval_id,))

    def expire_pending_for_task(self, task_id: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE approvals SET status='expired', handled=1 WHERE task_id=? AND status='pending'", (task_id,))

    # ── resources (subscription / API rate-limit cooldowns) ─────────────────

    def set_cooldown(self, name: str, until: float, reason: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO resources (name, cooldown_until, reason, updated_at) VALUES (?,?,?,?) "
                      "ON CONFLICT(name) DO UPDATE SET cooldown_until=excluded.cooldown_until, "
                      "reason=excluded.reason, updated_at=excluded.updated_at",
                      (name, until, reason, time.time()))
        self.log(None, "cooldown", f"{name} paused until {time.strftime('%H:%M', time.localtime(until))}: {reason}")

    def cooldown_until(self, name: str) -> float:
        with self._conn() as c:
            row = c.execute("SELECT cooldown_until FROM resources WHERE name=?", (name,)).fetchone()
        return float(row["cooldown_until"]) if row else 0.0

    def resources(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM resources").fetchall()]

    # ── meta (heartbeat etc.) ───────────────────────────────────────────────

    def set_meta(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO meta (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value, default=str)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default
