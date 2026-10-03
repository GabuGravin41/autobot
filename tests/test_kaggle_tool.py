"""
Offline tests for the kernel methods added to autobot/computer/kaggle_tool.py
(pull_kernel, kernel_status, kernel_output, push_kernel).

`Kaggle._get_api()` is mocked directly rather than patching the `kaggle`
package's internals — this sandbox doesn't have the `kaggle` pip package
installed (it's a real dependency of the user's actual machine, not of
these tests), and mocking at `_get_api()` means these tests don't care
either way. They verify: the right underlying KaggleApi method is called
with the right arguments, results are formatted as plain strings (what
computer_call's dispatch.py expects back), and push_kernel's
kernel-metadata.json precondition is enforced before ever touching the API.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autobot.computer.kaggle_tool import Kaggle


def _kaggle_with_mock_api() -> tuple[Kaggle, MagicMock]:
    k = Kaggle()
    mock_api = MagicMock()
    k._api = mock_api   # bypass _get_api()'s real KaggleApi() + authenticate()
    return k, mock_api


class TestPullKernel:
    def test_calls_kernels_pull_with_metadata(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        result = k.pull_kernel("user/my-kernel", str(tmp_path / "work"))
        api.kernels_pull.assert_called_once_with("user/my-kernel", str(tmp_path / "work"), metadata=True)
        assert "user/my-kernel" in result
        assert (tmp_path / "work").exists()

    def test_missing_kernel_raises(self):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(ValueError, match="kernel slug"):
            k.pull_kernel("", "./somewhere")
        api.kernels_pull.assert_not_called()


class TestKernelStatus:
    def test_returns_status_as_string(self):
        k, api = _kaggle_with_mock_api()
        api.kernels_status.return_value = "complete"
        result = k.kernel_status("user/my-kernel")
        assert result == "complete"
        api.kernels_status.assert_called_once_with("user/my-kernel")

    def test_missing_kernel_raises(self):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(ValueError):
            k.kernel_status("")


class TestKernelOutput:
    def test_uses_cli_in_utf8_child_process(self, tmp_path, monkeypatch):
        # The in-process API writes the kernel log with cp1252 on Windows and
        # crashes on progress-bar glyphs; the CLI is run with PYTHONUTF8=1.
        calls = []
        monkeypatch.setattr("autobot.computer.kaggle_tool._run_kaggle_cli",
                            lambda args, timeout=120: calls.append(args) or "")
        k, api = _kaggle_with_mock_api()
        out_dir = tmp_path / "out"
        result = k.kernel_output("user/my-kernel", str(out_dir))
        assert calls == [["kernels", "output", "user/my-kernel", "-p", str(out_dir), "-o"]]
        api.kernels_output.assert_not_called()
        assert out_dir.exists()
        assert "downloaded" in result

    def test_falls_back_to_api_when_cli_missing(self, tmp_path, monkeypatch):
        def no_cli(args, timeout=120):
            raise RuntimeError("kaggle CLI not found on PATH. Run: pip install kaggle")
        monkeypatch.setattr("autobot.computer.kaggle_tool._run_kaggle_cli", no_cli)
        k, api = _kaggle_with_mock_api()
        k.kernel_output("user/my-kernel", str(tmp_path / "o"))
        api.kernels_output.assert_called_once()

    def test_utf8_env_forces_utf8_in_child(self):
        from autobot.computer.kaggle_tool import _utf8_env
        env = _utf8_env()
        assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"


class TestPushKernel:
    def test_blocks_without_metadata_file(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(FileNotFoundError, match="kernel-metadata.json"):
            k.push_kernel(str(tmp_path))
        api.kernels_push.assert_not_called()   # never touches the API

    def test_pushes_when_metadata_present(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        (tmp_path / "kernel-metadata.json").write_text("{}")
        api.kernels_push.return_value = "Kernel version 3 successfully pushed"
        result = k.push_kernel(str(tmp_path))
        api.kernels_push.assert_called_once_with(str(tmp_path))
        assert "pushed" in result

    def test_empty_path_raises(self):
        k, api = _kaggle_with_mock_api()
        with pytest.raises(ValueError):
            k.push_kernel("")


class TestPushKernelLivenessVerification:
    """Sep 2026 Round 7: push_kernel now runs the 60-Second Liveness
    Verification Rule by default (see kaggle_watchdog.py's module
    docstring for the real incident this closes). These tests use a
    tmp_path-backed ledger and a no-op sleep_fn so they run instantly
    rather than actually blocking ~60s."""

    def _kaggle_with_metadata(self, tmp_path, kernel_id="user/my-kernel"):
        k, api = _kaggle_with_mock_api()
        k._ledger_path = tmp_path / "jobs.json"
        (tmp_path / "kernel-metadata.json").write_text(json.dumps({"id": kernel_id}))
        return k, api

    def test_running_status_reports_survived(self, tmp_path):
        k, api = self._kaggle_with_metadata(tmp_path)
        api.kernels_push.return_value = "Kernel version 2 successfully pushed"
        api.kernels_status.return_value = "running"
        result = k.push_kernel(str(tmp_path), sleep_fn=lambda s: None)
        assert "Liveness verified" in result
        assert "running" in result
        # Ledger persisted the job with the verified status.
        ledger = k._get_ledger()
        job = ledger.get("user/my-kernel")
        assert job is not None
        assert job.status == "running"
        assert job.liveness_verified is True

    def test_error_status_reports_failure_not_exception(self, tmp_path):
        # A dead-on-arrival kernel must be reported, not silently treated
        # as a successful push — this is the exact "optimistic launch
        # bias" failure mode from AUTOBOT_AUTONOMOUS_EXECUTION_CHALLENGES.md.
        k, api = self._kaggle_with_metadata(tmp_path)
        api.kernels_push.return_value = "Kernel version 3 successfully pushed"
        api.kernels_status.return_value = "error"
        result = k.push_kernel(str(tmp_path), sleep_fn=lambda s: None)
        assert "LIVENESS CHECK FAILED" in result
        ledger = k._get_ledger()
        assert ledger.get("user/my-kernel").status == "error"

    def test_no_id_in_metadata_skips_verification(self, tmp_path):
        # The original test above uses `{}` metadata (no "id") — must keep
        # working exactly as before: liveness verification is skipped, not
        # an error, and kernels_status is never called.
        k, api = _kaggle_with_mock_api()
        k._ledger_path = tmp_path / "jobs.json"
        (tmp_path / "kernel-metadata.json").write_text("{}")
        api.kernels_push.return_value = "Kernel version 1 successfully pushed"
        result = k.push_kernel(str(tmp_path), sleep_fn=lambda s: None)
        assert result == "Kernel version 1 successfully pushed"
        api.kernels_status.assert_not_called()

    def test_verify_liveness_false_skips_entirely(self, tmp_path):
        k, api = self._kaggle_with_metadata(tmp_path)
        api.kernels_push.return_value = "Kernel version 1 successfully pushed"
        result = k.push_kernel(str(tmp_path), verify_liveness=False)
        assert result == "Kernel version 1 successfully pushed"
        api.kernels_status.assert_not_called()


class TestPushKernelCapacityEnforcement:
    """Round 8: push_kernel() refuses to dispatch (no API call at all) once
    Kaggle's account-wide GPU/CPU slot limit would be exceeded — see
    kaggle_watchdog.py's KAGGLE_GPU_SLOT_LIMIT/KAGGLE_CPU_SLOT_LIMIT. This is
    what makes orchestrator_dispatch.py's concurrent multi-project dispatch
    safe against oversubscribing a quota shared across the whole account."""

    def _push_with_hardware(self, tmp_path, kernel_id, enable_gpu, sub="k"):
        k, api = _kaggle_with_mock_api()
        k._ledger_path = tmp_path / "jobs.json"
        work = tmp_path / sub
        work.mkdir(exist_ok=True)
        (work / "kernel-metadata.json").write_text(json.dumps({"id": kernel_id, "enable_gpu": enable_gpu}))
        api.kernels_push.return_value = "Kernel version 1 successfully pushed"
        api.kernels_status.return_value = "running"
        return k, api, str(work)

    def test_gpu_dispatch_registers_hardware_in_ledger(self, tmp_path):
        k, api, work = self._push_with_hardware(tmp_path, "user/gpu-job", True)
        k.push_kernel(work, sleep_fn=lambda s: None)
        assert k._get_ledger().get("user/gpu-job").hardware == "gpu"

    def test_cpu_dispatch_registers_hardware_in_ledger(self, tmp_path):
        k, api, work = self._push_with_hardware(tmp_path, "user/cpu-job", False)
        k.push_kernel(work, sleep_fn=lambda s: None)
        assert k._get_ledger().get("user/cpu-job").hardware == "cpu"

    def test_third_gpu_dispatch_refused_without_calling_api(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        ledger_path = tmp_path / "jobs.json"
        k._ledger_path = ledger_path

        # Fill both real GPU slots first via two separate pushes.
        for i in range(2):
            work = tmp_path / f"gpu{i}"
            work.mkdir()
            (work / "kernel-metadata.json").write_text(json.dumps({"id": f"user/gpu{i}", "enable_gpu": True}))
            api.kernels_push.return_value = f"Kernel {i} pushed"
            api.kernels_status.return_value = "running"
            k.push_kernel(str(work), sleep_fn=lambda s: None)

        # A third GPU push must be refused BEFORE touching the API.
        work3 = tmp_path / "gpu2"
        work3.mkdir()
        (work3 / "kernel-metadata.json").write_text(json.dumps({"id": "user/gpu2", "enable_gpu": True}))
        api.kernels_push.reset_mock()
        with pytest.raises(RuntimeError, match="at capacity"):
            k.push_kernel(str(work3), sleep_fn=lambda s: None)
        api.kernels_push.assert_not_called()

    def test_enforce_capacity_false_bypasses_the_check(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        k._ledger_path = tmp_path / "jobs.json"
        for i in range(3):  # over the GPU limit of 2
            work = tmp_path / f"gpu{i}"
            work.mkdir()
            (work / "kernel-metadata.json").write_text(json.dumps({"id": f"user/gpu{i}", "enable_gpu": True}))
            api.kernels_push.return_value = f"Kernel {i} pushed"
            api.kernels_status.return_value = "running"
            # Must not raise even past the real limit, since capacity
            # enforcement is explicitly disabled.
            k.push_kernel(str(work), sleep_fn=lambda s: None, enforce_capacity=False)

    def test_cpu_capacity_is_independent_of_gpu(self, tmp_path):
        k, api = _kaggle_with_mock_api()
        k._ledger_path = tmp_path / "jobs.json"
        for i in range(2):  # fill both GPU slots
            work = tmp_path / f"gpu{i}"
            work.mkdir()
            (work / "kernel-metadata.json").write_text(json.dumps({"id": f"user/gpu{i}", "enable_gpu": True}))
            api.kernels_push.return_value = f"pushed {i}"
            api.kernels_status.return_value = "running"
            k.push_kernel(str(work), sleep_fn=lambda s: None)

        # A CPU push must still succeed — GPU being full doesn't touch CPU capacity.
        work_cpu = tmp_path / "cpu0"
        work_cpu.mkdir()
        (work_cpu / "kernel-metadata.json").write_text(json.dumps({"id": "user/cpu0", "enable_gpu": False}))
        api.kernels_push.return_value = "cpu pushed"
        result = k.push_kernel(str(work_cpu), sleep_fn=lambda s: None)
        assert "Liveness verified" in result


class TestGetLeaderboard:
    """Rewritten Sep 2026 to shell out to `kaggle competitions leaderboard
    download` instead of the removed competition_view_leaderboard() Python
    method — see kaggle_tool.py's module docstring. Tests patch
    subprocess.run to simulate the CLI writing a real leaderboard CSV into
    the temp directory it's given, matching the exact header format found
    in competitions/*/leaderboard/*.csv on Dalton's real machine."""

    _REAL_HEADER = "Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames"

    def _fake_subprocess_run(self, csv_rows):
        def _run(cmd, **kwargs):
            # -p <dir> is always the second-to-last-ish arg; find it robustly.
            out_dir = cmd[cmd.index("-p") + 1]
            csv_path = Path(out_dir) / "comp-publicleaderboard-fake.csv"
            csv_path.write_text(self._REAL_HEADER + "\n" + "\n".join(csv_rows))
            result = MagicMock()
            result.returncode = 0
            result.stdout = "Leaderboard downloaded"
            result.stderr = ""
            return result
        return _run

    def test_parses_real_csv_shape(self, tmp_path):
        k, _ = _kaggle_with_mock_api()
        rows = [
            '1,111,"Team One","2026-09-22 14:38:36",0.92406,34,teamone',
            '2,222,"Team Two","2026-09-21 09:00:00",0.90001,10,teamtwo',
        ]
        with patch("subprocess.run", side_effect=self._fake_subprocess_run(rows)):
            result = k.get_leaderboard("some-competition")
        assert result[0] == {
            "rank": 1, "teamName": "Team One", "score": 0.92406,
            "submissionCount": 34, "lastSubmissionDate": "2026-09-22 14:38:36",
        }
        assert len(result) == 2

    def test_respects_top_n(self, tmp_path):
        k, _ = _kaggle_with_mock_api()
        rows = [f'{i},{i},"Team {i}","2026-09-22 00:00:00",0.9,{i},t{i}' for i in range(1, 30)]
        with patch("subprocess.run", side_effect=self._fake_subprocess_run(rows)):
            result = k.get_leaderboard("some-competition", top_n=5)
        assert len(result) == 5

    def test_empty_competition_raises(self):
        k, _ = _kaggle_with_mock_api()
        with pytest.raises(ValueError):
            k.get_leaderboard("")

    def test_cli_not_found_raises_actionable_error(self):
        k, _ = _kaggle_with_mock_api()
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="pip install kaggle"):
                k.get_leaderboard("some-competition")


class TestSubmitCodeCompetition:
    """The Code-Competition submission path — see Biohub Cell Tracking's
    real, proven CLI command in
    competitions/biohub_cell_tracking/THINKING_AND_DECISIONS.md section 4.4."""

    def test_calls_cli_with_kernel_and_version_binding(self, tmp_path):
        k, _ = _kaggle_with_mock_api()
        sub_file = tmp_path / "submission.csv"
        sub_file.write_text("id,pred\n1,0\n")

        captured = {}
        def _run(cmd, **kwargs):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = "Successfully submitted"
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=_run):
            result = k.submit_code_competition(
                "biohub-cell-tracking-during-development",
                "daltongabrielomondi/autobot-biohub-exp1-sota-repro",
                3,
                str(sub_file),
                "Autobot Exp 1",
            )
        cmd = captured["cmd"]
        assert cmd[:3] == ["kaggle", "competitions", "submit"]
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "biohub-cell-tracking-during-development"
        assert "-k" in cmd and cmd[cmd.index("-k") + 1] == "daltongabrielomondi/autobot-biohub-exp1-sota-repro"
        assert "-v" in cmd and cmd[cmd.index("-v") + 1] == "3"
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == str(sub_file)
        assert "Successfully submitted" in result

    def test_missing_file_raises_before_shelling_out(self, tmp_path):
        k, _ = _kaggle_with_mock_api()
        with patch("subprocess.run") as mock_run:
            with pytest.raises(FileNotFoundError):
                k.submit_code_competition("comp", "user/kernel", 1, str(tmp_path / "nope.csv"), "msg")
            mock_run.assert_not_called()


class TestListTopKernels:
    def test_parses_cli_table_output(self):
        k, _ = _kaggle_with_mock_api()
        fake_output = (
            "ref                                  title                   author       lastRunTime          totalVotes  \n"
            "------------------------------------  ----------------------  -----------  -------------------  ----------  \n"
            "user1/great-notebook                  Great Notebook          user1        2026-09-20 10:00:00  120         \n"
            "user2/other-notebook                  Other Notebook          user2        2026-09-19 08:00:00  80          \n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_output, stderr="")
            result = k.list_top_kernels("some-competition")
        assert len(result) == 2
        assert result[0]["ref"] == "user1/great-notebook"

    def test_empty_competition_raises(self):
        k, _ = _kaggle_with_mock_api()
        with pytest.raises(ValueError):
            k.list_top_kernels("")
