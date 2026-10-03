"""
Regression tests for the Kaggle status bug found Sep 2026.

kaggle 2.x returns ApiGetKernelSessionStatusResponse from kernels_status();
its str() is '{"status": "RUNNING", "failureMessage": null}'. The old
normalizer lower-cased that and checked `"fail" in s` first, so every real
status (RUNNING, COMPLETE, QUEUED...) read as "error". That produced the
"LIVENESS CHECK FAILED ... status=error" with no error log seen in five
competitions, and jobs silently falling out of GPU/CPU slot accounting.

These fakes reproduce the real object's shape without needing the kaggle
package installed.
"""
from __future__ import annotations

import enum
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autobot.computer.kaggle_watchdog import (
    KaggleJobLedger,
    _normalize_status,
    check_capacity,
    failure_message,
    poll_pending,
    refresh_for_capacity,
    verify_liveness,
)


class KernelWorkerStatus(enum.Enum):
    QUEUED = 0
    RUNNING = 1
    COMPLETE = 2
    ERROR = 3
    CANCEL_REQUESTED = 4
    CANCEL_ACKNOWLEDGED = 5
    NEW_SCRIPT = 6


class FakeStatusResponse:
    """Same attributes and str() as kagglesdk's ApiGetKernelSessionStatusResponse."""

    def __init__(self, status: KernelWorkerStatus, failure: str | None = None):
        self.status = status
        self.failure_message = failure

    def __str__(self):
        return json.dumps({"status": self.status.name, "failureMessage": self.failure_message})


@pytest.mark.parametrize("st,expected", [
    (KernelWorkerStatus.QUEUED, "queued"),
    (KernelWorkerStatus.NEW_SCRIPT, "queued"),
    (KernelWorkerStatus.RUNNING, "running"),
    (KernelWorkerStatus.COMPLETE, "complete"),
    (KernelWorkerStatus.ERROR, "error"),
    (KernelWorkerStatus.CANCEL_REQUESTED, "cancelled"),
    (KernelWorkerStatus.CANCEL_ACKNOWLEDGED, "cancelled"),
])
def test_real_response_object_normalizes_correctly(st, expected):
    assert _normalize_status(FakeStatusResponse(st)) == expected


def test_the_exact_string_that_broke_the_old_normalizer():
    assert _normalize_status('{"status": "RUNNING", "failureMessage": null}') == "running"
    assert _normalize_status('{"status": "COMPLETE", "failureMessage": ""}') == "complete"


@pytest.mark.parametrize("raw,expected", [
    ("running", "running"), ("KernelWorkerStatus.RUNNING", "running"), ("complete", "complete"),
    ("error", "error"), ("queued", "queued"), ({"status": "ERROR"}, "error"), (None, "unknown"),
    ("something odd", "unknown"),
])
def test_other_shapes(raw, expected):
    assert _normalize_status(raw) == expected


def test_failure_message_is_extracted():
    r = FakeStatusResponse(KernelWorkerStatus.ERROR, "TabPFNLicenseError: no license")
    assert failure_message(r) == "TabPFNLicenseError: no license"
    assert failure_message(FakeStatusResponse(KernelWorkerStatus.RUNNING)) is None


def test_liveness_with_real_objects_reports_running(tmp_path):
    ledger = KaggleJobLedger(tmp_path / "jobs.json")
    api = MagicMock()
    api.kernels_status.return_value = FakeStatusResponse(KernelWorkerStatus.RUNNING)
    result = verify_liveness(api, ledger, "u/k", sleep_fn=lambda s: None)
    assert result["survived_init"] is True
    assert result["status"] == "running"


def test_liveness_error_uses_api_failure_message(tmp_path):
    ledger = KaggleJobLedger(tmp_path / "jobs.json")
    api = MagicMock()
    api.kernels_status.return_value = FakeStatusResponse(KernelWorkerStatus.ERROR, "ModuleNotFoundError: tabpfn")
    result = verify_liveness(api, ledger, "u/k", sleep_fn=lambda s: None)
    assert result["survived_init"] is False
    assert "ModuleNotFoundError" in result["error_log"]


def test_single_transient_error_read_is_not_fatal(tmp_path):
    ledger = KaggleJobLedger(tmp_path / "jobs.json")
    api = MagicMock()
    api.kernels_status.side_effect = [
        FakeStatusResponse(KernelWorkerStatus.ERROR),
        FakeStatusResponse(KernelWorkerStatus.RUNNING),
    ]
    result = verify_liveness(api, ledger, "u/k", sleep_fn=lambda s: None)
    assert result["survived_init"] is True
    assert result["status"] == "running"


def test_poll_pending_does_not_mark_running_jobs_as_error(tmp_path):
    ledger = KaggleJobLedger(tmp_path / "jobs.json")
    ledger.register("u/k", hardware="gpu")
    ledger.update("u/k", "running")
    api = MagicMock()
    api.kernels_status.return_value = FakeStatusResponse(KernelWorkerStatus.RUNNING)
    assert poll_pending(api, ledger) == []
    assert ledger.get("u/k").status == "running"
    assert ledger.active_hardware_count("gpu") == 1


def test_refresh_restores_a_job_the_old_bug_marked_error(tmp_path):
    # Ledger says GPU is free (the job was wrongly recorded as "error", no log),
    # but it is really running. After refresh the slot must count again.
    ledger = KaggleJobLedger(tmp_path / "jobs.json")
    ledger.register("u/a", hardware="gpu"); ledger.update("u/a", "running")
    ledger.register("u/b", hardware="gpu"); ledger.update("u/b", "error")  # bogus, no error_log
    assert check_capacity(ledger, "gpu")[0] is True  # the old, wrong answer
    api = MagicMock()
    api.kernels_status.return_value = FakeStatusResponse(KernelWorkerStatus.RUNNING)
    changed = refresh_for_capacity(api, ledger, max_age_s=0)
    assert changed == 1
    assert ledger.get("u/b").status == "running"
    assert check_capacity(ledger, "gpu")[0] is False


def test_refresh_leaves_real_errors_alone(tmp_path):
    ledger = KaggleJobLedger(tmp_path / "jobs.json")
    ledger.register("u/dead", hardware="gpu")
    ledger.update("u/dead", "error", error_log="Traceback: real crash")
    api = MagicMock()
    refresh_for_capacity(api, ledger, max_age_s=0)
    api.kernels_status.assert_not_called()


def test_concurrent_pushes_cannot_oversubscribe_gpu(tmp_path):
    """Two pushes racing for the last GPU slot: exactly one may win."""
    from autobot.computer.kaggle_tool import Kaggle

    ledger_path = tmp_path / "jobs.json"
    seed = KaggleJobLedger(ledger_path)
    seed.register("u/already", hardware="gpu"); seed.update("u/already", "running")

    def make_dir(name):
        d = tmp_path / name
        d.mkdir()
        (d / "kernel-metadata.json").write_text(json.dumps({"id": f"u/{name}", "enable_gpu": True}))
        return str(d)

    dirs = [make_dir("k1"), make_dir("k2")]
    results: list[str] = []

    def push(path):
        k = Kaggle(ledger_path=ledger_path)
        api = MagicMock()
        api.kernels_status.return_value = FakeStatusResponse(KernelWorkerStatus.RUNNING)

        def slow_push(p):
            time.sleep(0.3)   # widen the race window
            return "pushed"
        api.kernels_push.side_effect = slow_push
        k._api = api
        try:
            k.push_kernel(path, verify_liveness=False)
            results.append("ok")
        except RuntimeError as e:
            results.append("refused" if "refusing" in str(e) else f"other: {e}")

    threads = [threading.Thread(target=push, args=(d,)) for d in dirs]
    for t in threads: t.start()
    for t in threads: t.join(10)
    assert sorted(results) == ["ok", "refused"], results
    final = KaggleJobLedger(ledger_path)
    assert final.active_hardware_count("gpu") == 2
