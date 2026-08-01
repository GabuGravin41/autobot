"""Behavioral test for the CLI approval-hang fix (human_gate.py).

Root cause: wait_for_approval() only ever unblocks via respond(), and the
ONLY caller of respond() outside this module is web/app.py's
POST /api/human_input/respond — which only exists while `autobot --server`
is running. A run started with `autobot.cli "task"` never starts that
server, so any gated action would silently wait out the full timeout
(default up to 300s) and auto-block, with genuinely no way to approve it.

This proves the fix actually unblocks fast via a terminal prompt when stdin
is interactive, and that non-interactive contexts are untouched (no new
hang risk introduced there).
"""
import asyncio
import sys
import time
import builtins
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


from autobot.agent import human_gate as hg  # noqa: E402


def test_non_interactive_unaffected():
    """stdin not a TTY -> no terminal prompt spawns; pure-timeout path unchanged."""
    async def run():
        with patch.object(sys.stdin, "isatty", return_value=False):
            start = time.monotonic()
            result = await hg.wait_for_approval("k-noninteractive", "test", timeout=0.3)
            elapsed = time.monotonic() - start
            return result, elapsed

    result, elapsed = asyncio.run(run())
    check("non-interactive: times out (no way to answer) -> blocked", result is False)
    check("non-interactive: actually waited out the timeout, not instant",
          elapsed >= 0.25, f"elapsed={elapsed:.2f}s")


def test_terminal_prompt_allows():
    """stdin IS a TTY, user types 'y' -> approval returns True FAST, not after timeout."""
    async def run():
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(builtins, "input", return_value="y"):
            start = time.monotonic()
            result = await hg.wait_for_approval("k-allow", "test allow", timeout=10.0)
            elapsed = time.monotonic() - start
            return result, elapsed

    result, elapsed = asyncio.run(run())
    check("terminal 'y' -> approval allowed", result is True)
    check("resolved fast via terminal, did not wait out the 10s timeout",
          elapsed < 2.0, f"elapsed={elapsed:.2f}s")


def test_terminal_prompt_blocks_on_no():
    """stdin IS a TTY, user types 'n' (or anything but y/yes) -> blocked."""
    async def run():
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(builtins, "input", return_value="n"):
            return await hg.wait_for_approval("k-block", "test block", timeout=10.0)

    check("terminal 'n' -> blocked (fail-closed default)", asyncio.run(run()) is False)


def test_terminal_input_error_fails_closed():
    """If input() raises (e.g. EOF from a weird non-interactive edge case), fail closed."""
    async def run():
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(builtins, "input", side_effect=EOFError()):
            start = time.monotonic()
            result = await hg.wait_for_approval("k-eof", "test eof", timeout=0.5)
            elapsed = time.monotonic() - start
            return result, elapsed

    result, elapsed = asyncio.run(run())
    # input() raising immediately should resolve via respond("block") fast,
    # not fall through to waiting out the full timeout.
    check("input() error treated as block, resolves fast (fail-closed)",
          result is False and elapsed < 0.4, f"result={result} elapsed={elapsed:.2f}s")


def test_dashboard_still_works_alongside_terminal():
    """Both paths active at once (interactive stdin + a dashboard respond()).

    Real input() blocks for actual human seconds, so a genuine dashboard
    click can easily arrive first. A mocked input() that returns instantly
    would make the terminal path win every time regardless of timing, which
    doesn't test anything real — so the mock sleeps briefly first, the way
    an actual slow-typing human would, giving the dashboard's respond() a
    real window to win the race.
    """
    def _slow_no(_prompt):
        time.sleep(0.3)
        return "n"

    async def run():
        with patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(builtins, "input", side_effect=_slow_no):
            task = asyncio.create_task(hg.wait_for_approval("k-race", "race", timeout=5.0))
            await asyncio.sleep(0.05)  # let wait_for_approval register the key first
            hg.respond("k-race", "allow")  # dashboard answers before the "slow human" does
            return await task

    check("dashboard respond() wins when it answers before the terminal does",
          asyncio.run(run()) is True)


def test_state_cleaned_up_after_resolution():
    """No leftover keys in the module-level dicts after a request resolves."""
    async def run():
        with patch.object(sys.stdin, "isatty", return_value=False):
            await hg.wait_for_approval("k-cleanup", "test", timeout=0.1)

    asyncio.run(run())
    check("_events cleaned up", "k-cleanup" not in hg._events)
    check("_responses cleaned up", "k-cleanup" not in hg._responses)
    check("_messages cleaned up", "k-cleanup" not in hg._messages)


test_non_interactive_unaffected()
test_terminal_prompt_allows()
test_terminal_prompt_blocks_on_no()
test_terminal_input_error_fails_closed()
test_dashboard_still_works_alongside_terminal()
test_state_cleaned_up_after_resolution()

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
