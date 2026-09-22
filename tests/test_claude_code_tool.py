"""
Offline tests for autobot/computer/claude_code_tool.py — the thin
Computer.claude_code wrapper around autobot/integrations/claude_code_bridge.py.

claude_code_bridge.run_headless() itself is already covered thoroughly by
tests/test_claude_code_bridge.py (argv construction, subprocess failure
modes, etc.). These tests only cover what this wrapper adds: turning the
bridge's {"ok","data","error"} dict into either a plain string (for
computer_call's dispatch.py to hand back to the LLM) or a raised
RuntimeError (for dispatch.py's existing exception handling to catch and
report).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from autobot.computer.claude_code_tool import ClaudeCode


class TestRun:
    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_success_extracts_result_field(self, mock_run):
        mock_run.return_value = {"ok": True, "data": {"result": "Refactored foo.py", "session_id": "s1"}, "error": ""}
        cc = ClaudeCode()
        out = cc.run("refactor foo.py")
        assert out == "Refactored foo.py"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_success_with_string_data(self, mock_run):
        mock_run.return_value = {"ok": True, "data": "plain text result", "error": ""}
        cc = ClaudeCode()
        out = cc.run("do a thing")
        assert out == "plain text result"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_failure_raises_runtime_error(self, mock_run):
        mock_run.return_value = {"ok": False, "data": None, "error": "claude CLI not found on PATH."}
        cc = ClaudeCode()
        with pytest.raises(RuntimeError, match="claude CLI not found"):
            cc.run("do a thing")

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_forwards_all_params(self, mock_run):
        mock_run.return_value = {"ok": True, "data": "ok", "error": ""}
        cc = ClaudeCode()
        cc.run(
            "edit foo.py",
            cwd="/work",
            permission_mode="acceptEdits",
            allowed_tools=["Read", "Edit"],
            continue_session=True,
            resume_session_id="s2",
            timeout=42,
        )
        _, kwargs = mock_run.call_args
        assert kwargs["prompt"] == "edit foo.py"
        assert kwargs["cwd"] == "/work"
        assert kwargs["permission_mode"] == "acceptEdits"
        assert kwargs["allowed_tools"] == ["Read", "Edit"]
        assert kwargs["continue_session"] is True
        assert kwargs["resume_session_id"] == "s2"
        assert kwargs["timeout"] == 42

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_default_permission_mode_is_plan(self, mock_run):
        mock_run.return_value = {"ok": True, "data": "ok", "error": ""}
        cc = ClaudeCode()
        cc.run("look at this")
        assert mock_run.call_args.kwargs["permission_mode"] == "plan"
