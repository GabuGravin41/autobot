"""
A small cross-process file lock (Windows + POSIX), stdlib only.

Atomic writes (temp file + rename) stop a crash from leaving a half-written
JSON file, but they do NOT stop two processes from racing a
read -> decide -> write sequence. Kaggle capacity is the concrete case: two
agents both read "1/2 GPU slots used", both decide there's room, both push,
and the account ends up at 3/2. Wrapping check-then-register in this lock
makes it one critical section across every process on the machine.

Usage:
    with FileLock(path_to_lockfile, timeout=30):
        ...critical section...
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# Reentrancy bookkeeping: {resolved lock path: [owning thread id, depth, file handle]}.
# A thread that already holds a lock may acquire it again (e.g. push_kernel holds
# the ledger lock and then calls ledger.register(), which locks too).
_HELD: dict[str, list] = {}
_HELD_GUARD = threading.Lock()


class LockTimeout(TimeoutError):
    pass


class FileLock:
    def __init__(self, path: str | Path, timeout: float = 30.0, poll: float = 0.1,
                 reentrant: bool = True) -> None:
        self.path = Path(path)
        self.reentrant = reentrant
        self.timeout = timeout
        self.poll = poll
        self._fh = None

    def _key(self) -> str:
        return str(self.path.resolve())

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = self._key()
        me = threading.get_ident()
        with _HELD_GUARD:
            held = _HELD.get(key)
            if held and held[0] == me and self.reentrant:
                held[1] += 1
                self._reentrant = True
                return
        self._reentrant = False
        deadline = time.monotonic() + self.timeout
        fh = open(self.path, "a+b")
        while True:
            try:
                _lock(fh)
                self._fh = fh
                if self.reentrant:
                    with _HELD_GUARD:
                        _HELD[key] = [me, 1, fh]
                return
            except OSError:
                if time.monotonic() >= deadline:
                    fh.close()
                    raise LockTimeout(f"Could not acquire lock {self.path} within {self.timeout}s")
                time.sleep(self.poll)

    def release(self) -> None:
        key = self._key()
        with _HELD_GUARD:
            held = _HELD.get(key)
            if getattr(self, "_reentrant", False):
                if held:
                    held[1] -= 1
                self._reentrant = False
                return
            if self.reentrant and held and held[0] == threading.get_ident():
                held[1] -= 1
                if held[1] > 0:
                    return
                _HELD.pop(key, None)
        if self._fh is None:
            return
        try:
            _unlock(self._fh)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _lock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
