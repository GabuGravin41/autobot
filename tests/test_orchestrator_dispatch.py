"""
Offline tests for autobot/agent/orchestrator_dispatch.py — the "run
multiple projects at once" capability (Round 8), built directly on
orchestrator_checkin.py's existing patterns (real ProjectRegistry backed
by tmp_path, bridge functions mocked at the claude_code_bridge/
antigravity_bridge level, plain asyncio.run() since pytest-asyncio isn't
part of this project's test setup).

The single/happy-path tests mirror test_orchestrator_checkin.py closely on
purpose — dispatch_on() and check_in_on() share almost their whole shape,
differing mainly in what prompt gets sent and whether permission_mode/
skip_permissions is allowed to be write-capable. The new material here is
TestDispatchOnAllConcurrency: actually proving projects run concurrently
(not just "the loop calls each one"), the semaphore bound is respected,
and one project's exception can't corrupt another's result.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

from autobot.agent.orchestrator_dispatch import DispatchResult, OrchestratorDispatch
from autobot.knowledge.project_registry import ProjectRegistry


def _run(coro):
    return asyncio.run(coro)


def _new_registry(tmp_path):
    return ProjectRegistry(projects_dir=tmp_path / "projects")


class TestDispatchOnMissingProject:
    def test_returns_error_result_without_raising(self, tmp_path):
        orchestrator = OrchestratorDispatch(registry=_new_registry(tmp_path))
        result = _run(orchestrator.dispatch_on("ghost", "do the thing"))
        assert result.ok is False
        assert "No tracked project named" in result.error


class TestDispatchOnRequiresInstruction:
    def test_empty_instruction_rejected_before_touching_registry(self, tmp_path):
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "claude_code", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)
        result = _run(orchestrator.dispatch_on("p", "   "))
        assert result.ok is False
        assert "instruction is required" in result.error


class TestDispatchOnClaudeCode:
    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_successful_dispatch_updates_registry(self, mock_run, tmp_path):
        mock_run.return_value = {
            "ok": True,
            "data": {"result": "implemented the login flow", "session_id": "sess-9"},
            "error": "",
        }
        registry = _new_registry(tmp_path)
        registry.register("seqoy", "/work/seqoy", "claude_code", "referral dashboard")
        orchestrator = OrchestratorDispatch(registry=registry)

        result = _run(orchestrator.dispatch_on("seqoy", "implement the login flow"))

        assert result.ok is True
        assert result.summary == "implemented the login flow"
        stored = registry.get("seqoy")
        assert stored.last_status_summary == "implemented the login flow"
        assert stored.session_id == "sess-9"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_default_permission_mode_is_plan(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "claude_code", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)

        _run(orchestrator.dispatch_on("p", "do something"))

        assert mock_run.call_args.kwargs["permission_mode"] == "plan"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_caller_can_explicitly_request_write_access(self, mock_run, tmp_path):
        # This is the whole point of the caller-supplied param — dispatch_on
        # itself never upgrades the mode, but a caller (a human running
        # `autobot --dispatch`) can ask for it explicitly, per project.
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "claude_code", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)

        _run(orchestrator.dispatch_on("p", "write the fix", permission_mode="acceptEdits"))

        assert mock_run.call_args.kwargs["permission_mode"] == "acceptEdits"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_resumes_prior_session_id(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": "sess-1"}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "claude_code", "goal")
        registry.update_session("p", "sess-1")
        orchestrator = OrchestratorDispatch(registry=registry)

        _run(orchestrator.dispatch_on("p", "continue"))

        assert mock_run.call_args.kwargs["resume_session_id"] == "sess-1"

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_bridge_failure_does_not_update_registry(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": False, "data": None, "error": "claude CLI not found on PATH."}
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "claude_code", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)

        result = _run(orchestrator.dispatch_on("p", "do it"))

        assert result.ok is False
        assert registry.get("p").last_checked_at is None


class TestDispatchOnAntigravity:
    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_default_skip_permissions_is_false(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"response": "ok", "conversation_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "antigravity", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)

        _run(orchestrator.dispatch_on("p", "do something"))

        assert mock_run.call_args.kwargs["skip_permissions"] is False

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_caller_can_explicitly_request_skip_permissions(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"response": "ok", "conversation_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "antigravity", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)

        _run(orchestrator.dispatch_on("p", "do something", skip_permissions=True))

        assert mock_run.call_args.kwargs["skip_permissions"] is True

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    def test_successful_dispatch_uses_response_field_and_updates_session(self, mock_run, tmp_path):
        mock_run.return_value = {
            "ok": True,
            "data": {"response": "scraped 40 leads", "conversation_id": "conv-3"},
            "error": "",
        }
        registry = _new_registry(tmp_path)
        registry.register("lead-gen", "/work/leadgen", "antigravity", "scrape leads")
        orchestrator = OrchestratorDispatch(registry=registry)

        result = _run(orchestrator.dispatch_on("lead-gen", "scrape today's leads"))

        assert result.ok is True
        assert result.summary == "scraped 40 leads"
        assert registry.get("lead-gen").session_id == "conv-3"


class TestUnrecognizedBackend:
    def test_handled_gracefully_not_a_crash(self, tmp_path):
        registry = _new_registry(tmp_path)
        registry.register("legacy", "/work/legacy", "claude_code", "goal")
        project = registry.get("legacy")
        project.backend = "gui_clicking_v1"
        registry._save(project)
        orchestrator = OrchestratorDispatch(registry=registry)

        result = _run(orchestrator.dispatch_on("legacy", "do it"))

        assert result.ok is False
        assert "Unrecognized backend" in result.error


class TestExceptionDuringDispatch:
    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_exception_converted_to_error_result_not_raised(self, mock_run, tmp_path):
        mock_run.side_effect = RuntimeError("subprocess exploded")
        registry = _new_registry(tmp_path)
        registry.register("p", "/work/p", "claude_code", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)

        result = _run(orchestrator.dispatch_on("p", "do it"))

        assert result.ok is False
        assert "subprocess exploded" in result.error


class TestDispatchOnAllConcurrency:
    """The actual "run multiple projects like OpenClaw" behavior:
    dispatch_on_all() must run projects CONCURRENTLY (not one at a time
    like check_in_on_all()'s sequential for-loop), bounded by a semaphore,
    with one project's failure isolated from the rest."""

    def test_empty_instructions_returns_empty_list(self, tmp_path):
        orchestrator = OrchestratorDispatch(registry=_new_registry(tmp_path))
        results = _run(orchestrator.dispatch_on_all({}))
        assert results == []

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_dispatches_every_named_project(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("a", "/work/a", "claude_code", "goal a")
        registry.register("b", "/work/b", "claude_code", "goal b")
        registry.register("c", "/work/c", "claude_code", "goal c")  # not in instructions — untouched
        orchestrator = OrchestratorDispatch(registry=registry)

        results = _run(orchestrator.dispatch_on_all({"a": "do a", "b": "do b"}))

        assert {r.project_name for r in results} == {"a", "b"}
        assert mock_run.call_count == 2

    def test_projects_actually_run_concurrently_not_sequentially(self, tmp_path):
        # Real proof, not just "the loop iterates": three projects each
        # "take" 0.2s (via a fake bridge with a real asyncio.sleep). If
        # dispatch_on_all ran them sequentially, three would take >=0.6s;
        # run concurrently under a semaphore of 3, all three should finish
        # in close to one 0.2s slot's worth of wall-clock time.
        registry = _new_registry(tmp_path)
        for name in ("a", "b", "c"):
            registry.register(name, f"/work/{name}", "claude_code", f"goal {name}")
        orchestrator = OrchestratorDispatch(registry=registry)

        def _sync_slow_run_headless(**kwargs):
            # asyncio.to_thread runs this synchronously in a worker thread —
            # emulate the same real delay with a plain blocking sleep.
            time.sleep(0.2)
            return {"ok": True, "data": {"result": f"done: {kwargs['prompt']}", "session_id": None}, "error": ""}

        with patch("autobot.integrations.claude_code_bridge.run_headless", side_effect=_sync_slow_run_headless):
            start = time.monotonic()
            results = _run(orchestrator.dispatch_on_all(
                {"a": "task a", "b": "task b", "c": "task c"}, max_concurrent=3,
            ))
            elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.ok for r in results)
        # Generous bound (should be ~0.2s) — this only needs to prove it's
        # nowhere near the ~0.6s a sequential for-loop would take.
        assert elapsed < 0.5, f"expected concurrent execution, took {elapsed:.2f}s"

    def test_semaphore_bounds_actual_concurrency(self, tmp_path):
        # Five projects, max_concurrent=2 — never more than 2 should be
        # "in flight" (inside the bridge call) at the same instant.
        registry = _new_registry(tmp_path)
        for i in range(5):
            registry.register(f"p{i}", f"/work/p{i}", "claude_code", f"goal {i}")
        orchestrator = OrchestratorDispatch(registry=registry)

        in_flight = 0
        max_in_flight = 0

        def _tracking_run_headless(**kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            time.sleep(0.1)
            in_flight -= 1
            return {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}

        with patch("autobot.integrations.claude_code_bridge.run_headless", side_effect=_tracking_run_headless):
            results = _run(orchestrator.dispatch_on_all(
                {f"p{i}": f"task {i}" for i in range(5)}, max_concurrent=2,
            ))

        assert len(results) == 5
        assert max_in_flight <= 2

    @patch("autobot.integrations.antigravity_bridge.run_headless")
    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_one_project_failure_does_not_corrupt_others(self, mock_cc, mock_ag, tmp_path):
        registry = _new_registry(tmp_path)
        registry.register("a", "/work/a", "claude_code", "goal a")
        registry.register("b", "/work/b", "antigravity", "goal b")
        registry.register("c", "/work/c", "claude_code", "goal c")

        mock_cc.return_value = {"ok": True, "data": {"result": "fine", "session_id": None}, "error": ""}
        mock_ag.side_effect = RuntimeError("agy crashed hard")

        orchestrator = OrchestratorDispatch(registry=registry)
        results = _run(orchestrator.dispatch_on_all({"a": "do a", "b": "do b", "c": "do c"}))

        by_name = {r.project_name: r for r in results}
        assert by_name["a"].ok is True
        assert by_name["c"].ok is True
        assert by_name["b"].ok is False
        assert "agy crashed hard" in by_name["b"].error

    @patch("autobot.integrations.claude_code_bridge.run_headless")
    def test_per_project_permission_mode_override(self, mock_run, tmp_path):
        mock_run.return_value = {"ok": True, "data": {"result": "ok", "session_id": None}, "error": ""}
        registry = _new_registry(tmp_path)
        registry.register("read-only-proj", "/work/r", "claude_code", "goal")
        registry.register("write-proj", "/work/w", "claude_code", "goal")
        orchestrator = OrchestratorDispatch(registry=registry)

        _run(orchestrator.dispatch_on_all(
            {"read-only-proj": "check something", "write-proj": "fix the bug"},
            permission_modes={"write-proj": "acceptEdits"},
        ))

        calls = {c.kwargs["cwd"]: c.kwargs["permission_mode"] for c in mock_run.call_args_list}
        assert calls["/work/r"] == "plan"
        assert calls["/work/w"] == "acceptEdits"


class TestDispatchResultToDict:
    def test_to_dict_shape(self):
        result = DispatchResult(project_name="p", backend="claude_code", ok=True, summary="done")
        assert result.to_dict() == {
            "project_name": "p",
            "backend": "claude_code",
            "ok": True,
            "summary": "done",
            "error": "",
        }
