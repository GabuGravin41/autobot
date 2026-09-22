"""
Offline tests for autobot/computer/antigravity_tool.py — the thin
Computer.antigravity wrapper around autobot/integrations/antigravity_bridge.py.

Mirrors tests/test_claude_code_tool.py deliberately closely. Covers only
what this wrapper adds beyond antigravity_bridge.run_headless() itself
(already covered by tests/test_antigravity_bridge.py): turning the
bridge's {"ok","data","error"} dict into a plain string or a raised
RuntimeError, and specifically the "response" vs "result" field-name
difference from claude_code_tool.py's equivalent extraction (the two
CLIs' JSON envelopes name the same concept differently — confirmed
against each CLI's own docs, not assumed to match).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from autobot.computer.antigravity_tool import Antigravity


class TestRun:
    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_success_extracts_response_field(self, mock_run):
        mock_run.return_value = {"ok": True, "data": {"response": "Implemented the feature", "status": "SUCCESS"}, "error": ""}
        ag = Antigravity()
        out = ag.run("implement the feature")
        assert out == "Implemented the feature"

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_success_with_string_data(self, mock_run):
        mock_run.return_value = {"ok": True, "data": "plain text result", "error": ""}
        ag = Antigravity()
        out = ag.run("do a thing")
        assert out == "plain text result"

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_falls_back_to_result_field_if_no_response_key(self, mock_run):
        # Defensive: if a future agy version's envelope ever used "result"
        # like Claude Code's does instead of "response", this still works
        # rather than silently stringifying the whole dict.
        mock_run.return_value = {"ok": True, "data": {"result": "fallback text"}, "error": ""}
        ag = Antigravity()
        out = ag.run("do a thing")
        assert out == "fallback text"

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_failure_raises_runtime_error(self, mock_run):
        mock_run.return_value = {"ok": False, "data": None, "error": "agy CLI not found on PATH."}
        ag = Antigravity()
        with pytest.raises(RuntimeError, match="agy CLI not found"):
            ag.run("do a thing")

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_forwards_all_params(self, mock_run):
        mock_run.return_value = {"ok": True, "data": "ok", "error": ""}
        ag = Antigravity()
        ag.run(
            "edit foo.py",
            cwd="/work",
            skip_permissions=True,
            model="gemini-3.8-flash-high",
            effort="high",
            agent="coder",
            continue_session=True,
            conversation_id="conv-1",
            timeout=42,
        )
        _, kwargs = mock_run.call_args
        assert kwargs["prompt"] == "edit foo.py"
        assert kwargs["cwd"] == "/work"
        assert kwargs["skip_permissions"] is True
        assert kwargs["model"] == "gemini-3.8-flash-high"
        assert kwargs["effort"] == "high"
        assert kwargs["agent"] == "coder"
        assert kwargs["continue_session"] is True
        assert kwargs["conversation_id"] == "conv-1"
        assert kwargs["timeout"] == 42

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_default_skip_permissions_is_false(self, mock_run):
        mock_run.return_value = {"ok": True, "data": "ok", "error": ""}
        ag = Antigravity()
        ag.run("look at this")
        assert mock_run.call_args.kwargs["skip_permissions"] is False
