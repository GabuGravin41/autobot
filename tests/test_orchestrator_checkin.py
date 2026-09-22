"""
Offline tests for autobot/agent/orchestrator_checkin.py.

Uses a REAL ProjectRegistry backed by tmp_path (not a mock) — this module's
whole job is registry-in, registry-out, so mocking the registry would mostly
test the mock. The two AI backends ARE mocked, at the bridge-function level
(claude_code_bridge.run_headless / antigravity_bridge.run_headless) rather
than the Computer.claude_code/.antigravity wrapper level, matching how the
module itself calls them (see its own docstring for why) and how
tests/test_antigravity_tool.py mocks the same bridge functions.

pytest-asyncio isn't part of this project's test setup (see
tests/test_unattended_approval.py / tests/test_core_loop.py) — async code
under test is driven with a plain `asyncio.run()` helper instead, same
convention followed here.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from autobot.agent.orchestrator_checkin import CheckInResult, OrchestratorCheckIn
from autobot.knowledge.project_registry import ProjectRegistry


def _run(coro):
    return asyncio.run(coro)


def _new_registry(tmp_path):
    return ProjectRegistry(projects_dir=tmp_path / "projects")


class TestCheckInOnMissingProject:
    def test_returns_error_result_without_raising(self, tmp_path):
        orchestrator = OrchestratorCheckIn(registry=_new_registry(tmp_path))
        result = _run(orchestrator.check_in_on("ghost"))
        assert result.ok is False
        assert "No tracked project named" in result.error
        assert result.project_name == "ghost"

    def test_error_lists_registered_projects(self, tmp_path):
        registry = _new_registry(tmp_path)
        registry.register("real-project", "/work/real", "claude_code", "goal")
        orchestrator = OrchestratorCheckIn(registry=registry)
        result = _run(orchestrator.check_in_on("ghost"))
        assert "real-project" in result.error


class TestCheckInOnClaudeCode:
    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_successful_check_in_updates_registry(self, mock_run, tmp_path):
        mock_run.return_value = {
            "ok": True,
            "data": {"result": "80% done, blocked on API key", "session_id": "sess-42"},
            "error": "",
        }
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "referral dashboard")
        orchestrator = OrchestratorCheckIn(registry=registry)

        result = _run(orchestrator.check_in_on("seqoy"))

        assert result.ok is True
        assert result.summary == "80% done, blocked on API key"
        assert result.backend == "claude_code"

        stored = registry.get("seqoy")
        assert stored.last_status_summary == "80% done, blocked on API key"
        assert stored.last_checked_at is not None
        assert stored.session_id == "sess-42"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_prompt_includes_intent_and_notes(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "build a referral dashboard")
        registry.add_intent_note("seqoy", "needs Twilio SMS webhooks")
        orchestrator = OrchestratorCheckIn(registry=registry)

        _run(orchestrator.check_in_on("seqoy"))

        prompt = mock_run.call_args.kwargs["prompt"]
        assert "build a referral dashboard" in prompt
        assert "needs Twilio SMS webhooks" in prompt
        assert "status read, not an instruction" in prompt

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_resumes_prior_session_id(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": "sess-1"}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "goal")
        registry.update_session("seqoy", "sess-1")
        orchestrator = OrchestratorCheckIn(registry=registry)

        _run(orchestrator.check_in_on("seqoy"))

        assert mock_run.call_args.kwargs["resume_session_id"] == "sess-1"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_permission_mode_never_forced_to_a_write_mode(self, mock_run, tmp_path):
        # The bridge call doesn't pass permission_mode explicitly — the
        # bridge's own default ("plan") is what makes this read-only. Pin
        # that no write-capable mode is ever threaded through here.
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "goal")
        orchestrator = OrchestratorCheckIn(registry=registry)

        _run(orchestrator.check_in_on("seqoy"))

        assert "permission_mode" not in mock_run.call_args.kwargs

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_same_session_id_in_and_out_does_not_crash(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": "sess-1"}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "goal")
        registry.update_session("seqoy", "sess-1")
        orchestrator = OrchestratorCheckIn(registry=registry)

        _run(orchestrator.check_in_on("seqoy"))

        assert registry.get("seqoy").session_id == "sess-1"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_bridge_failure_does_not_update_registry(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": False, "data": None, "error": "claude CLI not found on PATH."}
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "goal")
        orchestrator = OrchestratorCheckIn(registry=registry)

        result = _run(orchestrator.check_in_on("seqoy"))

        assert result.ok is False
        assert result.error == "claude CLI not found on PATH."
        assert registry.get("seqoy").last_checked_at is None  # unchanged


class TestCheckInOnAntigravity:
    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_successful_check_in_uses_response_field(self, mock_run, tmp_path):
        mock_run.return_value = {
            "ok": True,
            "data": {"response": "3 of 5 tasks done", "conversation_id": "conv-7", "status": "SUCCESS"},
            "error": "",
        }
        registry = _new_registry(tmp_path)
        registry.register("lead-gen", "/work/leadgen", "antigravity", "scrape leads")
        orchestrator = OrchestratorCheckIn(registry=registry)

        result = _run(orchestrator.check_in_on("lead-gen"))

        assert result.ok is True
        assert result.summary == "3 of 5 tasks done"
        assert result.backend == "antigravity"
        assert registry.get("lead-gen").session_id == "conv-7"

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_resumes_prior_conversation_id(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"response": "ok", "conversation_id": "conv-1"}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("lead-gen", "/work/leadgen", "antigravity", "goal")
        registry.update_session("lead-gen", "conv-1")
        orchestrator = OrchestratorCheckIn(registry=registry)

        _run(orchestrator.check_in_on("lead-gen"))

        assert mock_run.call_args.kwargs["conversation_id"] == "conv-1"

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_skip_permissions_never_passed(self, mock_run, tmp_path):
        # A check-in must never accidentally request write access.
        mock_run.return_value = {"ok": True, "data": {"response": "ok", "conversation_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("lead-gen", "/work/leadgen", "antigravity", "goal")
        orchestrator = OrchestratorCheckIn(registry=registry)

        _run(orchestrator.check_in_on("lead-gen"))

        assert "skip_permissions" not in mock_run.call_args.kwargs


class TestUnrecognizedBackend:
    def test_handled_gracefully_not_a_crash(self, tmp_path):
        # Simulate a registry file written by an older/incompatible schema
        # version — register() itself validates, so go around it and write
        # a raw record directly, the way a hand-edited or legacy-version
        # file could exist on disk.
        registry = _new_registry(tmp_path)
        registry.register("legacy", "/work/legacy", "claude_code", "goal")
        project = registry.get("legacy")
        project.backend = "gui_clicking_v1"  # not in KNOWN_BACKENDS
        registry._save(project)
        orchestrator = OrchestratorCheckIn(registry=registry)

        result = _run(orchestrator.check_in_on("legacy"))

        assert result.ok is False
        assert "Unrecognized backend" in result.error
        assert "gui_clicking_v1" in result.error


class TestExceptionDuringDispatch:
    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_exception_converted_to_error_result_not_raised(self, mock_run, tmp_path):
        mock_run.side_effect = RuntimeError("subprocess exploded")
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "goal")
        orchestrator = OrchestratorCheckIn(registry=registry)

        result = _run(orchestrator.check_in_on("seqoy"))

        assert result.ok is False
        assert "subprocess exploded" in result.error
        assert registry.get("seqoy").last_checked_at is None


class TestCheckInOnAll:
    def test_empty_registry_returns_empty_list(self, tmp_path):
        orchestrator = OrchestratorCheckIn(registry=_new_registry(tmp_path))
        results = _run(orchestrator.check_in_on_all())
        assert results == []

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_one_project_failure_does_not_stop_the_batch(self, mock_cc, mock_ag, tmp_path):
        registry = _new_registry(tmp_path)
        registry.register("a", "/work/a", "claude_code", "goal a")
        registry.register("b", "/work/b", "antigravity", "goal b")
        registry.register("c", "/work/c", "claude_code", "goal c")

        # "b" (antigravity) fails; "a" and "c" (claude_code) succeed.
        mock_cc.return_value = {"ok": True, "data": {"result": "fine", "session_id": None}, "error": ""}
        mock_ag.return_value = {"ok": False, "data": None, "error": "agy not found"}

        orchestrator = OrchestratorCheckIn(registry=registry)
        results = _run(orchestrator.check_in_on_all())

        assert len(results) == 3
        by_name = {r.project_name: r for r in results}
        assert by_name["a"].ok is True
        assert by_name["b"].ok is False
        assert by_name["c"].ok is True

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_checks_in_on_every_registered_project(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("a", "/work/a", "claude_code", "goal a")
        registry.register("b", "/work/b", "claude_code", "goal b")
        orchestrator = OrchestratorCheckIn(registry=registry)

        results = _run(orchestrator.check_in_on_all())

        assert mock_run.call_count == 2
        assert {r.project_name for r in results} == {"a", "b"}


class TestCheckInResultToDict:
    def test_to_dict_shape(self):
        result = CheckInResult(project_name="p", backend="claude_code", ok=True, summary="all good")
        d = result.to_dict()
        assert d == {
            "project_name": "p",
            "backend": "claude_code",
            "ok": True,
            "summary": "all good",
            "error": "",
        }
