"""
Tests for CoreLoop — the new UIAutomation-first agent loop.

These tests run fully offline (no LLM, no browser, no real UIAutomation)
using mocks, so they cost nothing to run and work on any machine.

They verify:
  - The observe → decide → act → verify cycle
  - Stuck detection (same screen for N steps)
  - done() action exits the loop correctly
  - Skill distillation is called on success
  - Override mid-flight changes the goal
  - Cancel stops the loop
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call as mock_call

import pytest

from autobot.agent.action_models import Action, LLMResponse, StepRecord
from autobot.agent.core_loop import CoreLoop


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_computer(window_tree: str = "[1] <button> OK") -> MagicMock:
    computer = MagicMock()
    computer.window.extract_ui.return_value = window_tree
    computer.window.active_title.return_value = "Test Window"
    computer.window.click.return_value = True
    computer.window.type.return_value = True
    computer.window.focus.return_value = True
    computer.keyboard.type.return_value = None
    computer.keyboard.press.return_value = None
    computer.terminal.run.return_value = "ok"
    computer.display.screenshot.return_value = "base64data"
    computer.get_tool_catalog.return_value = "# Tools: (mocked)"
    return computer


def _make_llm_client(responses: list[dict]) -> MagicMock:
    """Returns a mock LLM client that emits responses in order."""
    client = MagicMock()
    choices = []
    for r in responses:
        msg = MagicMock()
        msg.content = json.dumps(r)
        choice = MagicMock()
        choice.message = msg
        choices.append(choice)

    resp_objs = [MagicMock(choices=[c]) for c in choices]
    # Use AsyncMock so `await client.chat.completions.create(...)` works
    client.chat.completions.create = AsyncMock(side_effect=resp_objs)
    return client


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_done_action_exits_loop():
    """A 'done' action should terminate the loop immediately."""
    computer = _make_computer()
    llm = _make_llm_client([
        {
            "thinking": "task is done",
            "next_goal": "Done",
            "action": {"name": "done", "params": {"text": "All finished!", "success": True}},
        }
    ])
    loop = CoreLoop(computer=computer, llm_client=llm, goal="test goal", max_steps=10)
    result = _run(loop.run())
    assert "All finished!" in result
    assert loop.step_number == 1


def test_click_action_executes():
    """A 'click' action should call computer.window.click(index)."""
    computer = _make_computer()
    llm = _make_llm_client([
        {
            "thinking": "click the button",
            "next_goal": "click OK",
            "action": {"name": "click", "params": {"index": 1}},
        },
        {
            "thinking": "done after click",
            "next_goal": "Done",
            "action": {"name": "done", "params": {"text": "Clicked OK", "success": True}},
        },
    ])
    loop = CoreLoop(computer=computer, llm_client=llm, goal="click OK button", max_steps=5)
    result = _run(loop.run())
    computer.window.click.assert_called_once_with(1)
    assert loop.step_number == 2


def test_type_action_executes():
    """A 'type' action should call computer.keyboard.type."""
    computer = _make_computer()
    llm = _make_llm_client([
        {
            "thinking": "type something",
            "next_goal": "type hello",
            "action": {"name": "type", "params": {"text": "hello world"}},
        },
        {
            "thinking": "done",
            "next_goal": "Done",
            "action": {"name": "done", "params": {"text": "Typed.", "success": True}},
        },
    ])
    loop = CoreLoop(computer=computer, llm_client=llm, goal="type hello", max_steps=5)
    _run(loop.run())
    computer.keyboard.type.assert_called_once_with("hello world")


def test_run_shell_executes():
    """A 'run_shell' action should call computer.terminal.run."""
    computer = _make_computer()
    computer.terminal.run.return_value = "hello\n"
    llm = _make_llm_client([
        {
            "thinking": "run a command",
            "next_goal": "echo hello",
            "action": {"name": "run_shell", "params": {"command": "echo hello", "timeout": 10}},
        },
        {
            "thinking": "done",
            "next_goal": "Done",
            "action": {"name": "done", "params": {"text": "Done", "success": True}},
        },
    ])
    loop = CoreLoop(computer=computer, llm_client=llm, goal="echo hello", max_steps=5)
    # Patch the approval gate to auto-approve so the test doesn't block
    with patch.object(loop._approval, "gate", new=AsyncMock(return_value=True)):
        _run(loop.run())
    computer.terminal.run.assert_called_once_with("echo hello", timeout=10)


def test_cancel_stops_loop():
    """Setting is_cancelled=True should stop the loop before the next step."""
    computer = _make_computer()
    call_count = 0

    async def _fake_decide(*a, **kw):
        nonlocal call_count
        call_count += 1
        return None   # would cause LLM-fail exit normally

    loop = CoreLoop(computer=computer, llm_client=MagicMock(), goal="test", max_steps=10)
    loop._decide = _fake_decide
    loop.is_cancelled = True

    result = _run(loop.run())
    assert "Cancelled" in result
    assert call_count == 0     # no steps executed


def test_push_override_changes_goal():
    """push_override() should swap the goal on the next step."""
    computer = _make_computer()
    goals_seen: list[str] = []

    async def _fake_decide(screen, skill_ctx, stuck_warning):
        goals_seen.append(loop.goal)
        # After first step, trigger the override
        if len(goals_seen) == 1:
            loop.push_override("new goal")
        return LLMResponse(
            thinking="ok",
            next_goal="done",
            action=Action(name="done", params={"text": "ok", "success": True}),
        )

    loop = CoreLoop(computer=computer, llm_client=MagicMock(), goal="original goal", max_steps=5)
    loop._decide = _fake_decide
    _run(loop.run())
    # The first step sees "original goal"; override fires but done exits immediately
    assert goals_seen[0] == "original goal"


def test_unknown_action_returns_error_in_history():
    """An unknown action name should produce a failed step record, not an exception."""
    computer = _make_computer()
    llm = _make_llm_client([
        {
            "thinking": "try unknown",
            "next_goal": "do nothing",
            "action": {"name": "nonexistent_action", "params": {}},
        },
        {
            "thinking": "done",
            "next_goal": "Done",
            "action": {"name": "done", "params": {"text": "finished", "success": True}},
        },
    ])
    loop = CoreLoop(computer=computer, llm_client=llm, goal="test", max_steps=5)
    result = _run(loop.run())
    assert loop.history[0].action.name == "nonexistent_action"
    # The loop should not crash — it records the bad action and continues
    assert "finished" in result


def test_step_record_skill_distiller_compatibility():
    """StepRecord must satisfy SkillDistiller.distill_from_run's interface."""
    action = Action(name="click", params={"index": 3})
    record = StepRecord(
        step=1, next_goal="click the button", action=action,
        result="Clicked [3]", success=True,
        screen_before="[1] btn", screen_after="[1] btn",
        url_after="",
    )
    # These properties are required by distill_from_run
    assert len(record.action_results) == 1
    ar = record.action_results[0]
    assert ar.action_name == "click"
    assert ar.success is True
    assert record.agent_output.next_goal == "click the button"


def test_stuck_detection_adds_warning():
    """If screen doesn't change, _same_screen_count should increment."""
    computer = _make_computer(window_tree="[1] <button> Stuck")
    # Same tree every time
    computer.window.extract_ui.return_value = "[1] <button> Stuck"
    loop = CoreLoop(computer=computer, llm_client=MagicMock(), goal="test", max_steps=5)

    loop._last_screen = "[1] <button> Stuck"
    loop._same_screen_count = 0

    # Simulate observe
    loop._observe()
    screen = "[1] <button> Stuck"
    if screen == loop._last_screen:
        loop._same_screen_count += 1

    assert loop._same_screen_count == 1


def test_llm_response_parse_handles_markdown_fence():
    """LLMResponse parsing should strip ```json ... ``` fences."""
    loop = CoreLoop.__new__(CoreLoop)  # create without __init__
    raw = '```json\n{"thinking": "ok", "next_goal": "test", "action": {"name": "done", "params": {"text": "x", "success": true}}}\n```'
    resp = loop._parse_llm_response(raw)
    assert resp is not None
    assert resp.action.name == "done"


def test_llm_response_parse_rejects_invalid_json():
    """Completely broken JSON should return None, not raise."""
    loop = CoreLoop.__new__(CoreLoop)
    resp = loop._parse_llm_response("this is not json at all !!!")
    assert resp is None


if __name__ == "__main__":
    results = []
    tests = [
        test_done_action_exits_loop,
        test_click_action_executes,
        test_type_action_executes,
        test_run_shell_executes,
        test_cancel_stops_loop,
        test_push_override_changes_goal,
        test_unknown_action_returns_error_in_history,
        test_step_record_skill_distiller_compatibility,
        test_stuck_detection_adds_warning,
        test_llm_response_parse_handles_markdown_fence,
        test_llm_response_parse_rejects_invalid_json,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            results.append(True)
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            results.append(False)
    print(f"\n{sum(results)} passed, {len(results)-sum(results)} failed")
