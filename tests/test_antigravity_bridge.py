"""
Offline tests for autobot/integrations/antigravity_bridge.py.

run_cli is mocked; no real `agy` is invoked. Output shape follows the
documented headless JSON envelope (antigravity.google/docs/cli/headless/):
response, conversation_id, status, structured_output.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from autobot.integrations import antigravity_bridge
from autobot.integrations.cli_exec import CliResult

WHICH = "shutil.which"
RUN = "autobot.integrations.antigravity_bridge.run_cli"

OK = json.dumps({"response": "done", "conversation_id": "conv-1", "status": "SUCCESS"})


def _res(code=0, out="", err="", **kw):
    return CliResult(code, out, err, **kw)


def _argv(run, call=-1):
    return run.call_args_list[call][0][0]


class TestAvailability:
    @patch(WHICH, return_value="/usr/local/bin/agy")
    def test_available(self, _):
        assert antigravity_bridge.is_available() is True

    @patch(WHICH, return_value=None)
    def test_not_available(self, _):
        assert antigravity_bridge.is_available() is False


class TestRunHeadless:
    def test_empty_prompt_rejected(self):
        assert antigravity_bridge.run_headless(" ")["ok"] is False

    @patch(WHICH, return_value=None)
    def test_cli_not_installed(self, _):
        r = antigravity_bridge.run_headless("x")
        assert r["ok"] is False and r["error_class"] == "not_installed"

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_success_parses_json(self, run, _):
        run.return_value = _res(0, OK)
        r = antigravity_bridge.run_headless("x")
        assert r["ok"] is True and r["data"]["conversation_id"] == "conv-1"

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_print_timeout_passed_so_agy_does_not_kill_itself_at_5m(self, run, _):
        run.return_value = _res(0, OK)
        antigravity_bridge.run_headless("x", timeout=1800)
        args = _argv(run)
        assert args[args.index("--print-timeout") + 1] == "30m"

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_old_version_without_print_timeout_is_retried(self, run, _):
        run.side_effect = [_res(2, "", "unknown flag: --print-timeout"), _res(0, OK)]
        r = antigravity_bridge.run_headless("x")
        assert r["ok"] is True
        assert "--print-timeout" not in _argv(run, 1)

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_default_does_not_skip_permissions(self, run, _):
        run.return_value = _res(0, OK)
        antigravity_bridge.run_headless("x")
        assert "--dangerously-skip-permissions" not in _argv(run)

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_skip_permissions_true_forwarded(self, run, _):
        run.return_value = _res(0, OK)
        antigravity_bridge.run_headless("x", skip_permissions=True)
        assert "--dangerously-skip-permissions" in _argv(run)

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_model_effort_agent_forwarded(self, run, _):
        run.return_value = _res(0, OK)
        antigravity_bridge.run_headless("x", model="m", effort="high", agent="a")
        args = _argv(run)
        assert args[args.index("--model") + 1] == "m"
        assert args[args.index("--effort") + 1] == "high"
        assert args[args.index("--agent") + 1] == "a"

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_conversation_id_takes_priority_over_continue(self, run, _):
        run.return_value = _res(0, OK)
        antigravity_bridge.run_headless("x", conversation_id="c9", continue_session=True)
        args = _argv(run)
        assert args[args.index("--conversation") + 1] == "c9"
        assert "--continue" not in args

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_status_not_success_is_failure(self, run, _):
        run.return_value = _res(0, json.dumps({"status": "ERROR", "error": "quota exceeded"}))
        r = antigravity_bridge.run_headless("x")
        assert r["ok"] is False and "quota exceeded" in r["error"]
        assert r["error_class"] == "rate_limited"

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_nonzero_exit_surfaces_stderr(self, run, _):
        run.return_value = _res(1, "", "kaput")
        r = antigravity_bridge.run_headless("x")
        assert r["ok"] is False and "kaput" in r["error"]

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_timeout(self, run, _):
        run.return_value = _res(None, "", "", timed_out=True)
        r = antigravity_bridge.run_headless("x", timeout=5)
        assert r["ok"] is False and r["error_class"] == "timeout"

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_json_schema_written_to_file_and_cleaned_up(self, run, _):
        seen = {}

        def fake(argv, **kw):
            path = argv[argv.index("--json-schema") + 1]
            seen["path"] = path
            seen["content"] = json.loads(Path(path).read_text())
            return _res(0, json.dumps({"status": "SUCCESS", "structured_output": {"a": 1}}))
        run.side_effect = fake
        r = antigravity_bridge.run_headless("x", json_schema={"type": "object"})
        assert seen["content"] == {"type": "object"}
        assert not Path(seen["path"]).exists()
        assert r["data"]["structured_output"] == {"a": 1}

    @patch("autobot.integrations.antigravity_bridge.is_batch_shim", return_value=True)
    @patch("autobot.integrations.antigravity_bridge.resolve_exe", return_value=r"C:\npm\agy.cmd")
    @patch(WHICH, return_value=r"C:\npm\agy.cmd")
    @patch(RUN)
    def test_multiline_prompt_to_batch_shim_goes_via_file(self, run, _w, _r, _b, tmp_path):
        (tmp_path / ".git" / "info").mkdir(parents=True)
        seen = {}

        def fake(argv, **kw):
            short = argv[argv.index("-p") + 1]
            assert "\n" not in short
            name = short.split(".autobot/")[1].split(" ")[0]
            seen["text"] = (tmp_path / ".autobot" / name).read_text(encoding="utf-8")
            return _res(0, OK)
        run.side_effect = fake
        prompt = "Step 1: do this\nStep 2: 100% of tests must pass"
        antigravity_bridge.run_headless(prompt, cwd=str(tmp_path))
        assert seen["text"] == prompt
        assert ".autobot/" in (tmp_path / ".git" / "info" / "exclude").read_text()
        assert list((tmp_path / ".autobot").glob("task_*.md")) == []   # cleaned up

    @patch(WHICH, return_value="/usr/local/bin/agy")
    @patch(RUN)
    def test_nonexistent_cwd_reports_cwd_error(self, run, _, tmp_path):
        r = antigravity_bridge.run_headless("x", cwd=str(tmp_path / "nope"))
        assert r["ok"] is False and "does not exist" in r["error"]
        run.assert_not_called()
