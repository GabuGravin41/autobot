"""
Offline tests for autobot/computer/kaggle_watchdog.py — the persistent job
ledger and liveness-verification logic added Sep 2026 (Round 7) directly in
response to a real, documented incident (see that module's docstring): a
TabPFN kernel crashed 19 seconds after reporting RUNNING, and the agent
that dispatched it had already moved on and told the user it was fine,
because nothing was watching after its turn ended.

These tests use tmp_path for the ledger file and fake sleep_fn/time so
nothing actually blocks.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autobot.computer.kaggle_watchdog import (
    KAGGLE_CPU_SLOT_LIMIT,
    KAGGLE_GPU_SLOT_LIMIT,
    JobRecord,
    KaggleJobLedger,
    check_capacity,
    poll_pending,
    verify_liveness,
)


class TestKaggleJobLedger:
    def test_register_and_get(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/k1", competition="titanic", path="./work")
        job = ledger.get("user/k1")
        assert job is not None
        assert job.competition == "titanic"
        assert job.status == "dispatched"

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "jobs.json"
        ledger1 = KaggleJobLedger(path)
        ledger1.register("user/k1")
        ledger1.update("user/k1", "running")

        # A fresh instance reading the same file must see the same state —
        # this is the whole point: state survives a dead conversation.
        ledger2 = KaggleJobLedger(path)
        job = ledger2.get("user/k1")
        assert job is not None
        assert job.status == "running"

    def test_missing_file_starts_empty(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "does_not_exist.json")
        assert ledger.all() == []

    def test_corrupt_file_starts_fresh_without_crashing(self, tmp_path):
        path = tmp_path / "jobs.json"
        path.write_text("{not valid json")
        ledger = KaggleJobLedger(path)
        assert ledger.all() == []

    def test_pending_excludes_terminal_states(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/running-job")
        ledger.update("user/running-job", "running")
        ledger.register("user/done-job")
        ledger.update("user/done-job", "complete")
        ledger.register("user/dead-job")
        ledger.update("user/dead-job", "error")

        pending = {j.kernel for j in ledger.pending()}
        assert pending == {"user/running-job"}

    def test_history_is_capped(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/k1")
        for _ in range(80):
            ledger.update("user/k1", "running")
        assert len(ledger.get("user/k1").history) <= 50


class TestVerifyLiveness:
    def test_running_at_first_check_returns_survived(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        api = MagicMock()
        api.kernels_status.return_value = "running"
        result = verify_liveness(api, ledger, "user/k1", grace_checks=(30, 60), sleep_fn=lambda s: None)
        assert result["survived_init"] is True
        assert result["status"] == "running"
        assert result["checked_at"] == [30]  # stopped after first confirming check
        assert ledger.get("user/k1").liveness_verified is True

    def test_error_at_first_check_returns_failure_with_log(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        api = MagicMock()
        api.kernels_status.return_value = "error"

        def fetch_log(api, kernel):
            return "TabPFNLicenseError: no license found"

        result = verify_liveness(
            api, ledger, "user/k1", grace_checks=(30, 60),
            sleep_fn=lambda s: None, fetch_error_log=fetch_log,
        )
        assert result["survived_init"] is False
        assert result["status"] == "error"
        assert "TabPFNLicenseError" in result["error_log"]
        assert ledger.get("user/k1").status == "error"

    def test_queued_then_running_takes_second_check(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        api = MagicMock()
        api.kernels_status.side_effect = ["queued", "running"]
        result = verify_liveness(api, ledger, "user/k1", grace_checks=(30, 60), sleep_fn=lambda s: None)
        assert result["survived_init"] is True
        assert result["checked_at"] == [30, 60]

    def test_never_settles_is_reported_honestly(self, tmp_path):
        # Neither RUNNING/COMPLETE nor ERROR by the end of the grace period —
        # must not silently claim success.
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        api = MagicMock()
        api.kernels_status.return_value = "queued"
        result = verify_liveness(api, ledger, "user/k1", grace_checks=(30, 60), sleep_fn=lambda s: None)
        assert result["status"] == "unknown" or result["status"] == "queued"
        assert result["checked_at"] == [30, 60]

    def test_sleep_fn_called_with_correct_deltas(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        api = MagicMock()
        api.kernels_status.return_value = "running"
        sleeps = []
        verify_liveness(api, ledger, "user/k1", grace_checks=(30, 60), sleep_fn=sleeps.append)
        assert sleeps == [30]  # only one check needed since it's RUNNING immediately


class TestPollPending:
    def test_returns_only_changed_jobs(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/k1")
        ledger.update("user/k1", "running")
        ledger.register("user/k2")
        ledger.update("user/k2", "running")

        api = MagicMock()
        def status_for(kernel):
            return "complete" if kernel == "user/k1" else "running"
        api.kernels_status.side_effect = status_for

        changed = poll_pending(api, ledger)
        assert [j.kernel for j in changed] == ["user/k1"]
        assert ledger.get("user/k1").status == "complete"
        assert ledger.get("user/k2").status == "running"

    def test_skips_terminal_jobs(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/done")
        ledger.update("user/done", "complete")
        api = MagicMock()
        poll_pending(api, ledger)
        api.kernels_status.assert_not_called()

    def test_status_check_failure_does_not_crash_the_sweep(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/flaky")
        ledger.update("user/flaky", "running")
        api = MagicMock()
        api.kernels_status.side_effect = RuntimeError("network blip")
        changed = poll_pending(api, ledger)  # must not raise
        assert changed == []
        assert ledger.get("user/flaky").status == "running"  # unchanged, not corrupted


class TestHardwareCapacity:
    """The "Compute Resource Semaphore" from
    s6e9_ev_prediction/THINKING_AND_DECISIONS.md section 5.C — Kaggle's own
    hard account-wide caps (max 2 GPU kernels, max 4 CPU kernels), enforced
    locally before push_kernel() ever calls the real API, so running
    multiple Kaggle-competition projects concurrently (orchestrator_dispatch.py)
    can't silently oversubscribe a quota shared across the whole account."""

    def test_active_hardware_count_only_counts_pending_jobs_of_that_type(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/gpu1", hardware="gpu")
        ledger.update("user/gpu1", "running")
        ledger.register("user/gpu2", hardware="gpu")
        ledger.update("user/gpu2", "complete")  # terminal — must not count
        ledger.register("user/cpu1", hardware="cpu")
        ledger.update("user/cpu1", "running")

        assert ledger.active_hardware_count("gpu") == 1
        assert ledger.active_hardware_count("cpu") == 1

    def test_check_capacity_true_under_limit(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ledger.register("user/gpu1", hardware="gpu")
        ledger.update("user/gpu1", "running")
        ok, msg = check_capacity(ledger, "gpu")
        assert ok is True
        assert "1/2" in msg

    def test_check_capacity_false_at_gpu_limit(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        for i in range(KAGGLE_GPU_SLOT_LIMIT):
            ledger.register(f"user/gpu{i}", hardware="gpu")
            ledger.update(f"user/gpu{i}", "running")
        ok, msg = check_capacity(ledger, "gpu")
        assert ok is False
        assert "at capacity" in msg

    def test_check_capacity_false_at_cpu_limit(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        for i in range(KAGGLE_CPU_SLOT_LIMIT):
            ledger.register(f"user/cpu{i}", hardware="cpu")
            ledger.update(f"user/cpu{i}", "running")
        ok, _ = check_capacity(ledger, "cpu")
        assert ok is False

    def test_untracked_hardware_always_has_capacity(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        ok, msg = check_capacity(ledger, None)
        assert ok is True
        assert "not tracked" in msg

    def test_gpu_and_cpu_limits_are_independent(self, tmp_path):
        ledger = KaggleJobLedger(tmp_path / "jobs.json")
        for i in range(KAGGLE_GPU_SLOT_LIMIT):
            ledger.register(f"user/gpu{i}", hardware="gpu")
            ledger.update(f"user/gpu{i}", "running")
        # GPU is full, but CPU slots are untouched — must still have capacity.
        ok, _ = check_capacity(ledger, "cpu")
        assert ok is True
