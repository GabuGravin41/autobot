"""
Terminal Tool — Direct shell command execution for Autobot.

Gives the agent the ability to run shell commands and read output without
opening a terminal window, typing, waiting, and screenshotting.

Two modes:
  run()        — blocking: runs command, returns stdout+stderr when done
  start()      — non-blocking: starts a long process in background, returns PID
  output()     — read accumulated output from a background process
  wait()       — wait for a background process to finish
  kill()       — terminate a background process
  running()    — check if a background process is still alive

Usage (agent computer_call):
    computer.terminal.run("python --version")
    computer.terminal.run("git status", cwd="~/projects/myapp")
    computer.terminal.run("pip install numpy", timeout=120)

    pid = computer.terminal.start("python train.py --epochs 50", cwd="~/ml")
    computer.terminal.output(pid)     # check stdout so far
    computer.terminal.running(pid)    # still going?
    computer.terminal.wait(pid, timeout=3600)  # wait up to 1 hour
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 20_000   # truncate very long output to ~5k tokens
_DEFAULT_TIMEOUT = 60        # seconds for blocking run()


class _BackgroundProcess:
    """Tracks a single non-blocking subprocess."""
    def __init__(self, pid: int, proc: subprocess.Popen, log_path: Path) -> None:
        self.pid = pid
        self.proc = proc
        self.log_path = log_path
        self.started_at = time.time()
        self._output_offset = 0  # bytes already returned by output()


_LOG_RETENTION_DAYS = 7  # delete terminal logs older than this


def _cleanup_old_logs(log_dir: Path) -> None:
    """Delete log files older than _LOG_RETENTION_DAYS. Silent, best-effort."""
    try:
        cutoff = time.time() - _LOG_RETENTION_DAYS * 86400
        for f in log_dir.glob("proc_*.log"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception:
                pass
    except Exception:
        pass


class Terminal:
    """
    Shell command execution tool.

    Methods:
        run(command, cwd, timeout, env)  — run and return output (blocking)
        start(command, cwd, env)         — start background process, returns pid
        output(pid, tail)                — read new output from background process
        wait(pid, timeout)               — wait for background process to finish
        running(pid)                     — check if background process is alive
        kill(pid)                        — terminate background process
        processes()                      — list all tracked background processes
    """

    def __init__(self) -> None:
        self._procs: dict[int, _BackgroundProcess] = {}

    # ── Blocking execution ────────────────────────────────────────────────────

    def run(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        env: dict | None = None,
    ) -> str:
        """
        Run a shell command and return its output when it finishes.

        Use for quick commands (git, pip, python scripts under ~60s).
        For long-running processes use terminal.start() instead.

        Args:
            command: Shell command to run (passed to bash -c).
            cwd:     Working directory. Supports ~ expansion. Default: current dir.
            timeout: Max seconds to wait before killing (default 60).
            env:     Extra environment variables to set (merged with current env).

        Returns:
            Combined stdout + stderr, truncated if very long.
        """
        work_dir = Path(cwd).expanduser().resolve() if cwd else Path.cwd()
        merged_env = {**os.environ, **(env or {})}

        try:
            logger.info(f"💻 terminal.run: {command[:80]}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(work_dir),
                timeout=timeout,
                env=merged_env,
            )
            combined = ""
            if result.stdout:
                combined += result.stdout
            if result.stderr:
                combined += ("\n" if combined else "") + result.stderr

            if not combined.strip():
                combined = "(no output)"

            # Truncate and tag exit code
            if len(combined) > _MAX_OUTPUT_BYTES:
                combined = combined[:_MAX_OUTPUT_BYTES] + f"\n... (truncated, {len(combined)} bytes total)"

            exit_tag = f"\n[exit code: {result.returncode}]"
            return combined + exit_tag

        except subprocess.TimeoutExpired:
            return f"[TIMEOUT] Command exceeded {timeout}s limit: {command}\nUse terminal.start() for long-running commands."
        except FileNotFoundError as e:
            return f"[ERROR] Command not found: {e}"
        except Exception as e:
            return f"[ERROR] terminal.run failed: {e}"

    # ── Non-blocking execution ────────────────────────────────────────────────

    def start(
        self,
        command: str,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> str:
        """
        Start a long-running command in the background and return its PID.

        Use for: training scripts, servers, watch loops, anything that runs
        for more than a minute. Check progress with terminal.output(pid).

        Args:
            command: Shell command to run (passed to bash -c).
            cwd:     Working directory. Supports ~ expansion.
            env:     Extra environment variables.

        Returns:
            Status string with PID, e.g. "Started PID 12345: python train.py"
        """
        work_dir = Path(cwd).expanduser().resolve() if cwd else Path.cwd()
        merged_env = {**os.environ, **(env or {})}
        merged_env["PYTHONUNBUFFERED"] = "1"  # ensure Python output isn't buffered

        # Write output to a temp log file so we can tail it
        log_dir = Path.home() / ".autobot" / "terminal_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_logs(log_dir)  # prune logs older than 7 days
        log_path = log_dir / f"proc_{int(time.time())}.log"

        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(work_dir),
                    env=merged_env,
                    text=True,
                )
            bp = _BackgroundProcess(pid=proc.pid, proc=proc, log_path=log_path)
            self._procs[proc.pid] = bp
            logger.info(f"💻 terminal.start PID {proc.pid}: {command[:80]}")
            return (
                f"Started PID {proc.pid}: {command[:60]}\n"
                f"Output log: {log_path}\n"
                f"Check output: computer.terminal.output({proc.pid})\n"
                f"Check status: computer.terminal.running({proc.pid})"
            )
        except Exception as e:
            return f"[ERROR] terminal.start failed: {e}"

    def output(self, pid: int, tail: int = 100) -> str:
        """
        Read new output from a background process since last check.

        Calling this repeatedly gives you a rolling view of what the
        process has printed (like `tail -f`).

        Args:
            pid:  Process ID returned by terminal.start().
            tail: Max lines to return (default 100, newest lines).

        Returns:
            New output lines since last call, or status message.
        """
        bp = self._procs.get(pid)
        if not bp:
            return f"No tracked process with PID {pid}. Use terminal.processes() to see active ones."

        try:
            if not bp.log_path.exists():
                return "(no output yet)"

            content = bp.log_path.read_text(encoding="utf-8", errors="replace")
            # Return only new content since last read
            new_content = content[bp._output_offset:]
            bp._output_offset = len(content)

            if not new_content.strip():
                alive = bp.proc.poll() is None
                return f"(no new output — process {'still running' if alive else 'has finished'})"

            # Trim to last `tail` lines
            lines = new_content.splitlines()
            if len(lines) > tail:
                skipped = len(lines) - tail
                lines = [f"... ({skipped} earlier lines skipped)"] + lines[-tail:]

            status = "running" if bp.proc.poll() is None else f"finished (exit {bp.proc.poll()})"
            return f"[PID {pid} — {status}]\n" + "\n".join(lines)

        except Exception as e:
            return f"[ERROR] reading output for PID {pid}: {e}"

    def wait(self, pid: int, timeout: int = 300) -> str:
        """
        Wait for a background process to finish and return its final output.

        Args:
            pid:     Process ID returned by terminal.start().
            timeout: Max seconds to wait (default 300 = 5 minutes).

        Returns:
            Final output and exit code.
        """
        bp = self._procs.get(pid)
        if not bp:
            return f"No tracked process with PID {pid}."
        try:
            bp.proc.wait(timeout=timeout)
            # Read all remaining output
            bp._output_offset = 0  # reset to get full output
            full = self.output(pid, tail=200)
            return f"Process {pid} finished.\n{full}"
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT] PID {pid} still running after {timeout}s. Use terminal.output({pid}) to check progress."

    def running(self, pid: int) -> str:
        """
        Check if a background process is still running.

        Args:
            pid: Process ID returned by terminal.start().

        Returns:
            Status string: "running" | "finished (exit N)" | "not found"
        """
        bp = self._procs.get(pid)
        if not bp:
            return f"PID {pid} not found in tracked processes."
        rc = bp.proc.poll()
        if rc is None:
            elapsed = int(time.time() - bp.started_at)
            return f"PID {pid} still running ({elapsed}s elapsed)."
        return f"PID {pid} finished with exit code {rc}."

    def kill(self, pid: int) -> str:
        """
        Terminate a background process.

        Args:
            pid: Process ID returned by terminal.start().

        Returns:
            Confirmation string.
        """
        bp = self._procs.get(pid)
        if not bp:
            return f"PID {pid} not found."
        try:
            bp.proc.terminate()
            time.sleep(0.5)
            if bp.proc.poll() is None:
                bp.proc.kill()
            self._procs.pop(pid, None)
            return f"PID {pid} terminated."
        except Exception as e:
            return f"[ERROR] kill({pid}): {e}"

    def processes(self) -> str:
        """
        List all background processes started in this session.

        Returns:
            Formatted list of PIDs, commands, status and runtime.
        """
        if not self._procs:
            return "No background processes tracked in this session."
        lines = ["Background processes:\n"]
        for pid, bp in list(self._procs.items()):
            rc = bp.proc.poll()
            status = "running" if rc is None else f"exit {rc}"
            elapsed = int(time.time() - bp.started_at)
            lines.append(f"  PID {pid:<8} [{status:<12}] {elapsed}s  log: {bp.log_path.name}")
        return "\n".join(lines)
