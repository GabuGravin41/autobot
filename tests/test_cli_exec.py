"""Tests for autobot/integrations/cli_exec.py using real (POSIX) subprocesses."""
from __future__ import annotations

import os
import sys
import time

import pytest

from autobot.integrations.cli_exec import classify_error, is_batch_shim, run_cli

PY = sys.executable


def test_stdin_round_trip_with_newlines_and_specials():
    text = "line1\nline2 % & | \"quoted\" ünïcode ▉"
    r = run_cli([PY, "-c", "import sys; sys.stdout.write(sys.stdin.read())"], input_text=text, timeout=30)
    assert r.ok and r.stdout == text


def test_not_found_does_not_raise():
    r = run_cli(["definitely-not-a-real-binary-xyz"], timeout=5)
    assert r.not_found and not r.ok


def test_nonzero_exit_captured():
    r = run_cli([PY, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"], timeout=30)
    assert r.returncode == 3 and "bad" in r.stderr and not r.ok


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group check")
def test_timeout_kills_the_whole_tree(tmp_path):
    marker = tmp_path / "grandchild_alive"
    # Parent spawns a grandchild that would touch the marker after 3s.
    code = (
        "import subprocess, sys, time;"
        f"subprocess.Popen([sys.executable, '-c', 'import time,pathlib; time.sleep(3); pathlib.Path(r\"{marker}\").write_text(\"x\")']);"
        "time.sleep(60)"
    )
    t0 = time.time()
    r = run_cli([PY, "-c", code], timeout=1)
    assert r.timed_out and time.time() - t0 < 20
    time.sleep(3.5)
    assert not marker.exists(), "grandchild survived the timeout"


def test_child_gets_utf8_env():
    r = run_cli([PY, "-c", "import os; print(os.environ.get('PYTHONUTF8'))"], timeout=30)
    assert r.stdout.strip() == "1"


def test_is_batch_shim_only_on_windows():
    assert is_batch_shim(r"C:\x\claude.cmd") == (os.name == "nt")
    assert is_batch_shim("/usr/bin/claude") is False


@pytest.mark.parametrize("text,cls", [
    ("Claude usage limit reached. Resets at 5pm", "rate_limited"),
    ("Error code: 429 - Too Many Requests", "rate_limited"),
    ("RESOURCE_EXHAUSTED: quota", "rate_limited"),
    ("Invalid API key · Please run /login", "auth"),
    ("SyntaxError in file", "other"),
])
def test_classify_error(text, cls):
    assert classify_error(text) == cls
