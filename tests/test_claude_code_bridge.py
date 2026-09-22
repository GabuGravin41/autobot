"""
Offline tests for autobot/integrations/claude_code_bridge.py.

Every subprocess call is mocked — no real `claude` CLI is invoked and no
real API calls are made. These verify the wrapper's contract: correct argv
construction (especially the safety-relevant flags), never raising, and
correct handling of each CLI failure mode.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from autobot.integrations import claude_code_bridge


def _proc(returncode=0, stdout="", stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestAvailability:
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_available(self, _mock):
        assert claude_code_bridge.is_available() is True

    @patch("shutil.which", return_value=None)
    def test_not_available(self, _mock):
        assert claude_code_bridge.is_available() is False


class TestRunHeadless:
    def test_empty_prompt_rejected(self):
        result = claude_code_bridge.run_headless("")
        assert result["ok"] is False
        assert "prompt" in result["error"]

    def test_whitespace_only_prompt_rejected(self):
        result = claude_code_bridge.run_headless("   ")
        assert result["ok"] is False

    @patch("shutil.which", return_value=None)
    def test_cli_not_installed(self, _which):
        result = claude_code_bridge.run_headless("do something")
        assert result["ok"] is False
        assert "not found" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_success_parses_json(self, mock_run, _which):
        payload = {"result": "Refactored foo.py", "session_id": "abc123"}
        mock_run.return_value = _proc(0, stdout=json.dumps(payload))
        result = claude_code_bridge.run_headless("refactor foo.py")
        assert result["ok"] is True
        assert result["data"]["result"] == "Refactored foo.py"
        assert result["data"]["session_id"] == "abc123"

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_default_permission_mode_is_read_only(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("look at this code")
        args = mock_run.call_args[0][0]
        assert args[0] == "claude"
        idx = args.index("--permission-mode")
        assert args[idx + 1] == "plan"
        # headless mode can never answer an interactive prompt — this must
        # always be forced off regardless of permission_mode.
        assert "--permission-prompts" in args
        assert args[args.index("--permission-prompts") + 1] == "none"

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_write_capable_mode_forwarded_unmodified(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("edit foo.py", permission_mode="acceptEdits")
        args = mock_run.call_args[0][0]
        assert args[args.index("--permission-mode") + 1] == "acceptEdits"
        assert "acceptEdits" in claude_code_bridge.WRITE_CAPABLE_MODES

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_allowed_tools_forwarded_as_csv(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("x", allowed_tools=["Read", "Grep"])
        args = mock_run.call_args[0][0]
        idx = args.index("--allowedTools")
        assert args[idx + 1] == "Read,Grep"

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_no_allowed_tools_flag_when_omitted(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("x")
        args = mock_run.call_args[0][0]
        assert "--allowedTools" not in args

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_nonzero_exit_surfaces_stderr(self, mock_run, _which):
        mock_run.return_value = _proc(1, stdout="", stderr="authentication required")
        result = claude_code_bridge.run_headless("x")
        assert result["ok"] is False
        assert "authentication" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5))
    def test_timeout(self, _mock_run, _which):
        result = claude_code_bridge.run_headless("x", timeout=5)
        assert result["ok"] is False
        assert "timed out" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_malformed_json_output_does_not_fail_the_call(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="not json at all")
        result = claude_code_bridge.run_headless("x")
        assert result["ok"] is True
        assert result["data"]["result"] == "not json at all"
        assert "warning" in result["error"]

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_resume_session_flag(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("x", resume_session_id="sess-1")
        args = mock_run.call_args[0][0]
        assert args[args.index("--resume") + 1] == "sess-1"
        assert "--continue" not in args

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_continue_flag(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("x", continue_session=True)
        args = mock_run.call_args[0][0]
        assert "--continue" in args

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_resume_takes_priority_over_continue(self, mock_run, _which):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("x", continue_session=True, resume_session_id="sess-2")
        args = mock_run.call_args[0][0]
        assert "--resume" in args
        assert "--continue" not in args

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    @patch("subprocess.run")
    def test_cwd_forwarded_to_subprocess(self, mock_run, _which, tmp_path):
        mock_run.return_value = _proc(0, stdout="{}")
        claude_code_bridge.run_headless("x", cwd=str(tmp_path))
        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)
