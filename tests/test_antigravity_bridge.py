"""
Offline tests for autobot/integrations/antigravity_bridge.py.

Every subprocess call is mocked — no real `agy` CLI is invoked and no real
API calls are made. Mirrors tests/test_claude_code_bridge.py's structure
and coverage deliberately closely (see antigravity_bridge.py's own
docstring for why): argv construction (especially skip_permissions and
session-resume flags), never raising, correct handling of each CLI failure
mode, and the one thing this bridge checks that claude_code_bridge.py
doesn't need to — agy's JSON envelope carries its own `status` field
independent of the process exit code.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from autobot.integrations import antigravity_bridge


def _proc(returncode=0, stdout="", stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestAvailability:
    @patch("shutil.which", return_value="/usr/local/bin/agy")
    def test_available(self, _mock):
        assert antigravity_bridge.is_available() is True

    @patch("shutil.which", return_value=None)
    def test_not_available(self, _mock):
        assert antigravity_bridge.is_available() is False


class TestRunHeadless:
    def test_empty_prompt_rejected(self):
        result = antigravity_bridge.run_headless("")
        assert result["ok"] is False
        assert "prompt" in result["error"]

    def test_whitespace_only_prompt_rejected(self):
        result = antigravity_bridge.run_headless("   ")
        assert result["ok"] is False

    @patch("shutil.which", return_value=None)
    def test_cli_not_installed(self, _which):
        result = antigravity_bridge.run_headless("do something")
        assert result["ok"] is False
        assert "not found" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_success_parses_json(self, mock_run, _which):
        payload = {"status": "SUCCESS", "response": "Done", "conversation_id": "c1"}
        mock_run.return_value = _proc(0, stdout=json.dumps(payload))
        result = antigravity_bridge.run_headless("do something")
        assert result["ok"] is True
        assert result["data"]["response"] == "Done"
        assert result["data"]["conversation_id"] == "c1"

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_default_does_not_skip_permissions(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("look at this code")
        args = mock_run.call_args[0][0]
        assert args[0] == "agy"
        assert "-p" in args
        assert "--dangerously-skip-permissions" not in args

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_skip_permissions_true_forwarded(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("edit something", skip_permissions=True)
        args = mock_run.call_args[0][0]
        assert "--dangerously-skip-permissions" in args

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_output_format_json_always_set(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("x")
        args = mock_run.call_args[0][0]
        idx = args.index("--output-format")
        assert args[idx + 1] == "json"

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_model_effort_agent_forwarded(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("x", model="gemini-3.8-flash-high", effort="high", agent="coder")
        args = mock_run.call_args[0][0]
        assert args[args.index("--model") + 1] == "gemini-3.8-flash-high"
        assert args[args.index("--effort") + 1] == "high"
        assert args[args.index("--agent") + 1] == "coder"

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_no_optional_flags_when_omitted(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("x")
        args = mock_run.call_args[0][0]
        assert "--model" not in args
        assert "--effort" not in args
        assert "--agent" not in args

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_conversation_id_flag(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("x", conversation_id="conv-1")
        args = mock_run.call_args[0][0]
        assert args[args.index("--conversation") + 1] == "conv-1"
        assert "--continue" not in args

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_continue_flag(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("x", continue_session=True)
        args = mock_run.call_args[0][0]
        assert "--continue" in args

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_conversation_id_takes_priority_over_continue(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("x", continue_session=True, conversation_id="conv-2")
        args = mock_run.call_args[0][0]
        assert "--conversation" in args
        assert "--continue" not in args

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_nonzero_exit_surfaces_stderr(self, mock_run, _which):
        mock_run.return_value = _proc(1, stdout="", stderr="authentication required")
        result = antigravity_bridge.run_headless("x")
        assert result["ok"] is False
        assert "authentication" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="agy", timeout=5))
    def test_timeout(self, _mock_run, _which):
        result = antigravity_bridge.run_headless("x", timeout=5)
        assert result["ok"] is False
        assert "timed out" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_malformed_json_output_does_not_fail_the_call(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="not json at all")
        result = antigravity_bridge.run_headless("x")
        assert result["ok"] is True
        assert result["data"]["result"] == "not json at all"
        assert "warning" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_cwd_forwarded_to_subprocess(self, mock_run, _which, tmp_path):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS"}))
        antigravity_bridge.run_headless("x", cwd=str(tmp_path))
        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_nonexistent_cwd_reports_cwd_error_not_cli_missing(self, mock_run, _which, tmp_path):
        # Regression test: subprocess.run(cwd=...) raises FileNotFoundError
        # for a missing directory — the SAME exception type the CLI-not-
        # found path below catches — so without an explicit up-front check,
        # a stale/deleted project working_dir (very plausible: see
        # project_registry.py's working_dir field) surfaced as "agy CLI not
        # found on PATH", actively misleading a caller who has agy
        # installed just fine. subprocess.run must never even be reached.
        missing = str(tmp_path / "does-not-exist")
        result = antigravity_bridge.run_headless("x", cwd=missing)
        assert result["ok"] is False
        assert "cwd" in result["error"].lower()
        assert "does not exist" in result["error"].lower()
        assert "CLI not found" not in result["error"]
        mock_run.assert_not_called()

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_cwd_that_is_a_file_reports_cwd_error(self, mock_run, _which, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("x")
        result = antigravity_bridge.run_headless("x", cwd=str(f))
        assert result["ok"] is False
        assert "cwd" in result["error"].lower()
        mock_run.assert_not_called()

    # ── status field — the one behavior claude_code_bridge.py doesn't need,
    # since Claude Code's JSON envelope has no equivalent per-turn status
    # separate from the process exit code. agy's docs are explicit that a
    # clean exit (0) can still carry status CANCELED/INTERRUPTED/WAITING/etc,
    # which is a real distinct outcome a caller checking only returncode
    # would silently miss. ──────────────────────────────────────────────────

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_clean_exit_with_non_success_status_is_reported_as_failure(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "INTERRUPTED", "response": ""}))
        result = antigravity_bridge.run_headless("x")
        assert result["ok"] is False
        assert "INTERRUPTED" in result["error"]
        # The parsed envelope is still surfaced, not thrown away, so a
        # caller can inspect exactly what happened rather than just "it failed".
        assert result["data"]["status"] == "INTERRUPTED"

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_status_success_is_ok(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout=json.dumps({"status": "SUCCESS", "response": "done"}))
        result = antigravity_bridge.run_headless("x")
        assert result["ok"] is True

    @patch("shutil.which", return_value="/usr/local/bin/agy")
    @patch("subprocess.run")
    def test_missing_status_field_does_not_fail(self, mock_run, _which):
        # Malformed-but-valid-JSON output with no status key at all — don't
        # crash or misreport ok=False just because the field is absent;
        # treat "no status" as "nothing to disagree with the exit code about".
        mock_run.return_value = _proc(0, stdout=json.dumps({"response": "done, no status field"}))
        result = antigravity_bridge.run_headless("x")
        assert result["ok"] is True
