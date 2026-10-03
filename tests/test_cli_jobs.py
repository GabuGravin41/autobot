"""
Offline tests for autobot/cli.py's `--jobs` command (_show_jobs()) — the
one-shot Kaggle ledger dump for when `autobot --server`'s background
watchdog isn't running. No real kaggle package or credentials needed:
_show_jobs() is written to degrade gracefully (falls back to last-known
state) when authentication isn't available, which is exactly the case in
this sandbox and exactly the case this test exploits.
"""
from __future__ import annotations

from autobot.cli import _show_jobs
from autobot.computer.kaggle_watchdog import KaggleJobLedger


def test_empty_ledger_prints_helpful_message(tmp_path, monkeypatch, capsys):
    _show_jobs()
    out = capsys.readouterr().out
    assert "No Kaggle jobs recorded yet" in out


def test_populated_ledger_prints_each_job(tmp_path, monkeypatch, capsys):
    ledger = KaggleJobLedger()   # default path = $AUTOBOT_HOME (isolated per test by conftest)
    ledger.register("user/k1", competition="titanic")
    ledger.update("user/k1", "complete")
    ledger.register("user/k2", competition="titanic")
    ledger.update("user/k2", "error", error_log="boom")
    ledger.register("user/k3", competition="titanic")
    ledger.update("user/k3", "running")  # non-terminal -> triggers the live-poll attempt

    _show_jobs()
    out = capsys.readouterr().out
    assert "user/k1" in out and "status=complete" in out
    assert "user/k2" in out and "status=error" in out
    assert "boom" in out
    # No kaggle package/credentials in this sandbox — must degrade gracefully,
    # never raise, and say so rather than silently pretending it polled.
    assert "could not poll live status" in out
