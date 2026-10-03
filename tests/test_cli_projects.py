"""
Offline tests for autobot/cli.py's multi-project orchestrator commands
(--register-project, --list-projects, --check-in, --dispatch,
--dispatch-all) — the CLI wiring added Round 8 that makes the
already-built, already-tested ProjectRegistry / OrchestratorCheckIn /
OrchestratorDispatch classes actually reachable from a real run, closing
the gap found via grep: neither class had a single production caller
before this.

ProjectRegistry() with no args stores under Path.cwd() / "autobot" /
"knowledge" / "projects", so every test here monkeypatch.chdir(tmp_path)
to keep the real project's on-disk registry untouched. Async orchestrator
methods (check_in_on/_all, dispatch_on/_all) are monkeypatched directly
rather than exercising the real headless-CLI bridges — those bridges
already have their own test coverage (test_claude_code_bridge.py,
test_antigravity_bridge.py, test_orchestrator_dispatch.py,
test_orchestrator_checkin.py); this file's job is only to prove the CLI
layer calls them correctly and prints something useful.
"""
from __future__ import annotations

import json

import pytest

from autobot.cli import (
    _check_in,
    _dispatch,
    _dispatch_all,
    _list_projects,
    _register_project,
)


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class TestRegisterProject:
    def test_missing_args_exits_with_message(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _register_project("seqoy", None, None, None)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "--project-dir" in out

    def test_invalid_backend_exits_with_message(self, capsys, tmp_path):
        with pytest.raises(SystemExit) as exc:
            _register_project("seqoy", str(tmp_path), "chatgpt", "build the thing")
        assert exc.value.code == 1
        assert "Unknown backend" in capsys.readouterr().out

    def test_success_registers_and_prints_confirmation(self, capsys, tmp_path):
        _register_project("seqoy", str(tmp_path), "claude_code", "ship the EV predictor")
        out = capsys.readouterr().out
        assert "Registered 'seqoy'" in out
        assert "ship the EV predictor" in out

        from autobot.knowledge.project_registry import ProjectRegistry
        project = ProjectRegistry().get("seqoy")
        assert project is not None
        assert project.backend == "claude_code"


class TestListProjects:
    def test_empty_registry_prints_helpful_message(self, capsys):
        _list_projects()
        out = capsys.readouterr().out
        assert "No tracked projects yet" in out
        assert "--register-project" in out

    def test_registered_project_is_listed(self, capsys, tmp_path):
        _register_project("seqoy", str(tmp_path), "antigravity", "predict EV demand")
        capsys.readouterr()  # discard registration output

        _list_projects()
        out = capsys.readouterr().out
        assert "seqoy" in out
        assert "antigravity" in out
        assert "predict EV demand" in out
        assert "never checked in" in out


class TestCheckIn:
    def test_unregistered_project_reports_failure(self, capsys):
        _check_in("ghost-project")
        out = capsys.readouterr().out
        assert "[FAIL] ghost-project" in out
        assert "No tracked project named" in out

    def test_named_project_prints_ok_summary(self, capsys, tmp_path, monkeypatch):
        _register_project("seqoy", str(tmp_path), "claude_code", "ship it")
        capsys.readouterr()

        from autobot.agent.orchestrator_checkin import CheckInResult, OrchestratorCheckIn

        async def fake_check_in_on(self, project_name, timeout=300.0):
            return CheckInResult(project_name="seqoy", backend="claude_code", ok=True, summary="70% done")

        monkeypatch.setattr(OrchestratorCheckIn, "check_in_on", fake_check_in_on)

        _check_in("seqoy")
        out = capsys.readouterr().out
        assert "[OK] seqoy (claude_code)" in out
        assert "70% done" in out

    def test_no_name_checks_in_on_all(self, capsys, tmp_path, monkeypatch):
        from autobot.agent.orchestrator_checkin import CheckInResult, OrchestratorCheckIn

        async def fake_check_in_on_all(self, timeout=300.0):
            return [
                CheckInResult(project_name="a", backend="claude_code", ok=True, summary="fine"),
                CheckInResult(project_name="b", backend="antigravity", ok=False, summary="", error="timed out"),
            ]

        monkeypatch.setattr(OrchestratorCheckIn, "check_in_on_all", fake_check_in_on_all)

        _check_in(None)
        out = capsys.readouterr().out
        assert "[OK] a (claude_code)" in out and "fine" in out
        assert "[FAIL] b (antigravity)" in out and "timed out" in out

    def test_no_tracked_projects_says_so(self, capsys, monkeypatch):
        from autobot.agent.orchestrator_checkin import OrchestratorCheckIn

        async def fake_check_in_on_all(self, timeout=300.0):
            return []

        monkeypatch.setattr(OrchestratorCheckIn, "check_in_on_all", fake_check_in_on_all)

        _check_in(None)
        out = capsys.readouterr().out
        assert "No tracked projects" in out


class TestDispatch:
    def test_unregistered_project_reports_failure(self, capsys):
        _dispatch("ghost-project", "do the thing")
        out = capsys.readouterr().out
        assert "[FAIL] ghost-project" in out

    def test_success_prints_ok_summary(self, capsys, tmp_path, monkeypatch):
        _register_project("seqoy", str(tmp_path), "claude_code", "ship it")
        capsys.readouterr()

        from autobot.agent.orchestrator_dispatch import DispatchResult, OrchestratorDispatch

        async def fake_dispatch_on(self, project_name, instruction, timeout=600.0, permission_mode="plan", skip_permissions=False):
            assert instruction == "add a retry loop"
            return DispatchResult(project_name="seqoy", backend="claude_code", ok=True, summary="added it")

        monkeypatch.setattr(OrchestratorDispatch, "dispatch_on", fake_dispatch_on)

        _dispatch("seqoy", "add a retry loop")
        out = capsys.readouterr().out
        assert "[OK] seqoy (claude_code)" in out
        assert "added it" in out


class TestDispatchAll:
    def test_missing_file_exits(self, capsys, tmp_path):
        with pytest.raises(SystemExit) as exc:
            _dispatch_all(str(tmp_path / "does_not_exist.json"))
        assert exc.value.code == 1
        assert "Could not read" in capsys.readouterr().out

    def test_empty_instructions_prints_message_without_dispatching(self, capsys, tmp_path):
        instructions_file = tmp_path / "instructions.json"
        instructions_file.write_text("{}", encoding="utf-8")

        _dispatch_all(str(instructions_file))
        out = capsys.readouterr().out
        assert "lists no projects" in out

    def test_flat_instructions_dispatch_concurrently(self, capsys, tmp_path, monkeypatch):
        instructions_file = tmp_path / "instructions.json"
        instructions_file.write_text(
            json.dumps({"project-a": "do X", "project-b": "do Y"}), encoding="utf-8"
        )

        from autobot.agent.orchestrator_dispatch import DispatchResult, OrchestratorDispatch

        captured = {}

        async def fake_dispatch_on_all(self, instructions, timeout=600.0, max_concurrent=None,
                                        permission_modes=None, skip_permissions=None):
            captured["instructions"] = instructions
            captured["permission_modes"] = permission_modes
            captured["skip_permissions"] = skip_permissions
            return [
                DispatchResult(project_name="project-a", backend="claude_code", ok=True, summary="done a"),
                DispatchResult(project_name="project-b", backend="antigravity", ok=False, summary="", error="crashed"),
            ]

        monkeypatch.setattr(OrchestratorDispatch, "dispatch_on_all", fake_dispatch_on_all)

        _dispatch_all(str(instructions_file))
        out = capsys.readouterr().out
        assert captured["instructions"] == {"project-a": "do X", "project-b": "do Y"}
        assert "[OK] project-a (claude_code)" in out and "done a" in out
        assert "[FAIL] project-b (antigravity)" in out and "crashed" in out

    def test_nested_form_passes_permission_overrides_through(self, capsys, tmp_path, monkeypatch):
        instructions_file = tmp_path / "instructions.json"
        instructions_file.write_text(
            json.dumps({
                "instructions": {"project-b": "fix the bug"},
                "permission_modes": {"project-b": "acceptEdits"},
                "skip_permissions": {},
            }),
            encoding="utf-8",
        )

        from autobot.agent.orchestrator_dispatch import DispatchResult, OrchestratorDispatch

        captured = {}

        async def fake_dispatch_on_all(self, instructions, timeout=600.0, max_concurrent=None,
                                        permission_modes=None, skip_permissions=None):
            captured["permission_modes"] = permission_modes
            return [DispatchResult(project_name="project-b", backend="claude_code", ok=True, summary="fixed")]

        monkeypatch.setattr(OrchestratorDispatch, "dispatch_on_all", fake_dispatch_on_all)

        _dispatch_all(str(instructions_file))
        assert captured["permission_modes"] == {"project-b": "acceptEdits"}
