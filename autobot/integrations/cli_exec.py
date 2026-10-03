"""
Run an external CLI (claude, agy, kaggle, git, latexmk...) safely on Windows.

Three Windows problems the old bridges walked into:

1. `subprocess.run(["claude", ...])` does not honor PATHEXT. If the CLI
   was installed by npm it is `claude.cmd`, which `shutil.which()` finds
   (so is_available() said True) but CreateProcess does not — every call
   then failed with FileNotFoundError. We resolve the full path first.
2. A `.cmd`/`.bat` shim runs through cmd.exe, which truncates arguments at
   the first newline and expands %VARS% even inside quotes. Long, multi-line
   prompts must not travel as command-line arguments to a batch shim. The
   bridges send them on stdin (claude) or via a prompt file (agy) instead.
3. On timeout, killing the direct child leaves the real worker (node, or a
   grandchild) running. We kill the whole process tree.

Never uses shell=True.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Sequence

# Every child process started through run_cli, so a stopping daemon can take
# its workers down with it (children run in their own process group and would
# otherwise outlive it).
_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()


def kill_all_active() -> int:
    """Kill every process tree started by run_cli that is still running."""
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    for p in procs:
        if p.poll() is None:
            _kill_tree(p)
    return len(procs)


@dataclass
class CliResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    not_found: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.not_found


def resolve_exe(name: str) -> str | None:
    """Full path of an executable, honoring PATHEXT (.exe, .cmd, .bat) on Windows."""
    return shutil.which(name)


def is_batch_shim(path: str | None) -> bool:
    return bool(path) and os.name == "nt" and path.lower().endswith((".cmd", ".bat"))


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_cli(
    argv: Sequence[str],
    *,
    input_text: str | None = None,
    cwd: str | None = None,
    timeout: float = 600.0,
    env: dict[str, str] | None = None,
) -> CliResult:
    """Run argv (argv[0] is resolved via PATH/PATHEXT). Never raises."""
    if not argv:
        return CliResult(None, "", "empty argv", not_found=True)
    exe = resolve_exe(argv[0]) or (argv[0] if os.path.isfile(argv[0]) else None)
    if exe is None:
        return CliResult(None, "", f"{argv[0]} not found on PATH", not_found=True)

    popen_kwargs: dict = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    child_env = dict(os.environ if env is None else env)
    child_env.setdefault("PYTHONUTF8", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        proc = subprocess.Popen(
            [exe, *argv[1:]],
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=child_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except FileNotFoundError:
        return CliResult(None, "", f"{argv[0]} not found", not_found=True)
    except Exception as e:  # permissions, bad cwd, ...
        return CliResult(None, "", f"{type(e).__name__}: {e}")

    with _ACTIVE_LOCK:
        _ACTIVE.add(proc)
    try:
        out, err = proc.communicate(input=input_text, timeout=timeout)
        return CliResult(proc.returncode, out or "", err or "")
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = "", ""
        return CliResult(None, out or "", (err or "") + f"\n[timed out after {timeout:.0f}s]", timed_out=True)
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(proc)


_RATE_LIMIT_MARKERS = (
    "usage limit", "rate limit", "rate_limit", "429", "too many requests",
    "limit reached", "quota", "resource_exhausted", "overloaded", "try again later",
)
_AUTH_MARKERS = ("401", "authentication", "not logged in", "login required", "invalid api key",
                 "unauthorized", "please log in", "run /login", "credentials")


def classify_error(text: str) -> str:
    """Coarse error class used for scheduling decisions: rate_limited | auth | other."""
    t = (text or "").lower()
    if any(m in t for m in _RATE_LIMIT_MARKERS):
        return "rate_limited"
    if any(m in t for m in _AUTH_MARKERS):
        return "auth"
    return "other"


_UNKNOWN_FLAG_RE = __import__("re").compile(
    r"(?:unknown option|unknown flag|unrecognized option|unrecognized arguments?|no such option)[:\s'\"]*(--[A-Za-z0-9][\w-]*)",
    __import__("re").IGNORECASE,
)


def unknown_flag(text: str) -> str | None:
    """The flag a CLI rejected as unknown, if the error says so."""
    m = _UNKNOWN_FLAG_RE.search(text or "")
    return m.group(1) if m else None


def strip_flag(argv: list[str], flag: str, value_flags: set[str]) -> list[str]:
    """argv without `flag` (and its value, if it takes one)."""
    out, skip = [], False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == flag:
            skip = flag in value_flags
            continue
        out.append(tok)
    return out
