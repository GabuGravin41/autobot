"""
Offline tests for autobot/integrations/claude_code_bridge.py.

run_cli (the Windows-safe subprocess runner) is mocked — no real `claude`
CLI is invoked. These verify the wrapper's contract: argv construction
(especially safety flags), prompt on stdin, never raising, and handling of
each failure mode. The JSON output shape matches real Claude Code 2.1
output (checked against a live `claude -p --output-format json` run).
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from autobot.integrations import claude_code_bridge
from autobot.integrations.cli_exec import CliResult

WHICH = "shutil.which"
RUN = "autobot.integrations.claude_code_bridge.run_cli"

REAL_OK = json.dumps({
    "type": "result", "subtype": "success", "is_error": False,
    "result": "Done.", "session_id": "sess-1", "num_turns": 3,
})


def _res(code=0, out="", err="", **kw):
    return CliResult(code, out, err, **kw)


def _argv(mock_run, call=-1):
    return mock_run.call_args_list[call][0][0]


class TestAvailability:
    @patch(WHICH, return_value="/usr/local/bin/claude")
    def test_available(self, _):
        assert claude_code_bridge.is_available() is True

    @patch(WHICH, return_value=None)
    def test_not_available(self, _):
        assert claude_code_bridge.is_available() is False


class TestRunHeadless:
    def test_empty_prompt_rejected(self):
        r = claude_code_bridge.run_headless("  ")
        assert r["ok"] is False and "prompt" in r["error"]

    @patch(WHICH, return_value=None)
    def test_cli_not_installed(self, _):
        r = claude_code_bridge.run_headless("x")
        assert r["ok"] is False and r["error_class"] == "not_installed"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_success_parses_real_shape(self, run, _):
        run.return_value = _res(0, REAL_OK)
        r = claude_code_bridge.run_headless("fix the bug")
        assert r["ok"] is True
        assert r["data"]["session_id"] == "sess-1"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_prompt_goes_on_stdin_not_argv(self, run, _):
        run.return_value = _res(0, REAL_OK)
        prompt = "line one\nline two with % and & and \"quotes\""
        claude_code_bridge.run_headless(prompt)
        assert run.call_args.kwargs["input_text"] == prompt
        assert prompt not in _argv(run)

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_default_permission_mode_is_read_only(self, run, _):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x")
        args = _argv(run)
        assert args[args.index("--permission-mode") + 1] == "plan"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_write_capable_mode_forwarded_unmodified(self, run, _):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x", permission_mode="acceptEdits")
        args = _argv(run)
        assert args[args.index("--permission-mode") + 1] == "acceptEdits"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_allowed_and_disallowed_tools(self, run, _):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x", allowed_tools=["Read", "Edit"],
                                        disallowed_tools=["Bash(git push*)"])
        args = _argv(run)
        assert args[args.index("--allowedTools") + 1] == "Read,Edit"
        assert args[args.index("--disallowedTools") + 1] == "Bash(git push*)"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_no_tool_flags_when_omitted(self, run, _):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x")
        args = _argv(run)
        for flag in ("--allowedTools", "--disallowedTools", "--tools", "--json-schema"):
            assert flag not in args

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_tools_empty_disables_all_tools(self, run, _):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x", tools=[])
        args = _argv(run)
        assert args[args.index("--tools") + 1] == ""

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_json_schema_and_structured_output(self, run, _):
        out = json.dumps({"is_error": False, "result": "{}", "structured_output": {"move": "wait"}})
        run.return_value = _res(0, out)
        schema = {"type": "object", "properties": {"move": {"type": "string"}}}
        r = claude_code_bridge.run_headless("x", json_schema=schema)
        args = _argv(run)
        assert json.loads(args[args.index("--json-schema") + 1]) == schema
        assert r["data"]["structured_output"] == {"move": "wait"}

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_old_version_without_permission_prompts_flag_is_retried(self, run, _):
        run.side_effect = [
            _res(1, "", "error: unknown option '--permission-prompts'"),
            _res(0, REAL_OK),
        ]
        r = claude_code_bridge.run_headless("x")
        assert r["ok"] is True
        assert "--permission-prompts" in _argv(run, 0)
        assert "--permission-prompts" not in _argv(run, 1)

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_old_version_drops_each_unknown_flag(self, run, _):
        run.side_effect = [
            _res(1, "", "error: unknown option '--json-schema'"),
            _res(1, "", "error: unknown option '--no-session-persistence'"),
            _res(0, REAL_OK),
        ]
        r = claude_code_bridge.run_headless("x", json_schema={"type": "object"}, no_session_persistence=True)
        assert r["ok"] is True
        final = _argv(run, 2)
        assert "--json-schema" not in final and "--no-session-persistence" not in final
        assert '{"type": "object"}' not in final           # the flag's value went too

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_safety_flags_are_never_dropped(self, run, _):
        run.return_value = _res(1, "", "error: unknown option '--disallowedTools'")
        r = claude_code_bridge.run_headless("x", permission_mode="acceptEdits", disallowed_tools=["Bash(git push*)"])
        assert r["ok"] is False and run.call_count == 1

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_nonzero_exit_surfaces_stderr(self, run, _):
        run.return_value = _res(1, "", "boom")
        r = claude_code_bridge.run_headless("x")
        assert r["ok"] is False and "boom" in r["error"]

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_usage_limit_is_classified_as_rate_limited(self, run, _):
        out = json.dumps({"is_error": True, "result": "Claude usage limit reached. Your limit will reset at 5pm"})
        run.return_value = _res(0, out)
        r = claude_code_bridge.run_headless("x")
        assert r["ok"] is False
        assert r["error_class"] == "rate_limited"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_timeout(self, run, _):
        run.return_value = _res(None, "", "", timed_out=True)
        r = claude_code_bridge.run_headless("x", timeout=5)
        assert r["ok"] is False and r["error_class"] == "timeout"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_malformed_json_output_does_not_fail_the_call(self, run, _):
        run.return_value = _res(0, "plain text answer")
        r = claude_code_bridge.run_headless("x")
        assert r["ok"] is True and r["data"]["result"] == "plain text answer"

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_resume_takes_priority_over_continue(self, run, _):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x", continue_session=True, resume_session_id="sess-2")
        args = _argv(run)
        assert args[args.index("--resume") + 1] == "sess-2"
        assert "--continue" not in args

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_continue_flag(self, run, _):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x", continue_session=True)
        assert "--continue" in _argv(run)

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_cwd_forwarded(self, run, _, tmp_path):
        run.return_value = _res(0, REAL_OK)
        claude_code_bridge.run_headless("x", cwd=str(tmp_path))
        assert run.call_args.kwargs["cwd"] == str(tmp_path)

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_nonexistent_cwd_reports_cwd_error_not_cli_missing(self, run, _, tmp_path):
        r = claude_code_bridge.run_headless("x", cwd=str(tmp_path / "missing"))
        assert r["ok"] is False and "does not exist" in r["error"]
        assert "not found" not in r["error"]
        run.assert_not_called()

    @patch(WHICH, return_value="/usr/local/bin/claude")
    @patch(RUN)
    def test_cwd_that_is_a_file_reports_cwd_error(self, run, _, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x")
        r = claude_code_bridge.run_headless("x", cwd=str(f))
        assert r["ok"] is False and "cwd" in r["error"]
        run.assert_not_called()
