import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autobot.agent.models import (
    ActionModel,
    ActionResult,
    AgentOutput,
    ClickAction,
    NavigateAction,
    ProxyAssessment,
    StepHistoryEntry,
)
from autobot.agent.step_verifier import StepVerifier, VerificationResult
from autobot.learning.trace_logger import TraceLogger
from autobot.perception.manager import PerceptionManager, PerceptionSnapshot


def test_proxy_assessment_models():
    """Verify ProxyAssessment structure and integration into AgentOutput."""
    pa = ProxyAssessment(
        target_element="button[12] 'Sign In'",
        expected_state_delta="URL will change to /dashboard",
        active_window_focus="Chrome - Login",
        risk_label="safe",
    )
    assert pa.risk_label == "safe"
    assert "Sign In" in pa.target_element

    out = AgentOutput(
        thinking="Need to log in",
        proxy_assessment=pa,
        evaluation_previous_goal="Initialized session",
        memory="On login page",
        next_goal="Click login button",
        action=[ActionModel(click=ClickAction(index=12))],
    )
    assert out.proxy_assessment is not None
    assert out.proxy_assessment.expected_state_delta == "URL will change to /dashboard"

    # Test serialization round-trip
    dumped = out.model_dump()
    assert dumped["proxy_assessment"]["target_element"] == "button[12] 'Sign In'"
    restored = AgentOutput(**dumped)
    assert restored.proxy_assessment.risk_label == "safe"


def test_step_verifier_skipped_when_empty():
    verifier = StepVerifier()
    res = verifier.verify(
        proxy_assessment=None,
        action_results=[ActionResult(action_name="wait", success=True)],
        url_before="https://example.com",
        url_after="https://example.com",
    )
    assert res.passed is True
    assert res.label == "SKIPPED"


def test_step_verifier_action_failed():
    verifier = StepVerifier()
    pa = ProxyAssessment(expected_state_delta="URL will change to /dashboard")
    res = verifier.verify(
        proxy_assessment=pa,
        action_results=[
            ActionResult(action_name="click", success=False, error="Element not found")
        ],
        url_before="https://example.com/login",
        url_after="https://example.com/login",
    )
    assert res.passed is False
    assert res.label == "FAILED"
    assert "Element not found" in res.reason


def test_step_verifier_url_change_success():
    verifier = StepVerifier()
    pa = ProxyAssessment(expected_state_delta="URL will change to /dashboard")
    res = verifier.verify(
        proxy_assessment=pa,
        action_results=[ActionResult(action_name="click", success=True)],
        url_before="https://example.com/login",
        url_after="https://example.com/dashboard",
    )
    assert res.passed is True
    assert res.label == "VERIFIED"
    assert "URL changed as predicted" in res.reason


def test_step_verifier_silent_fail():
    verifier = StepVerifier()
    pa = ProxyAssessment(expected_state_delta="Page will navigate to /settings")
    res = verifier.verify(
        proxy_assessment=pa,
        action_results=[ActionResult(action_name="click", success=True)],
        url_before="https://example.com/home",
        url_after="https://example.com/home",  # URL did not change!
    )
    assert res.passed is False
    assert res.label == "SILENT_FAIL"
    assert "URL stayed" in res.reason


def test_step_verifier_keyword_match():
    verifier = StepVerifier()
    pa = ProxyAssessment(expected_state_delta="Command output shows download complete")
    res = verifier.verify(
        proxy_assessment=pa,
        action_results=[
            ActionResult(
                action_name="run_command",
                success=True,
                extracted_content="Status: download complete successfully in 3s",
            )
        ],
        url_before="about:blank",
        url_after="about:blank",
    )
    assert res.passed is True
    assert res.label == "VERIFIED"
    assert "Predicted keywords found" in res.reason


def test_step_history_entry_verification_text():
    out = AgentOutput(
        thinking="test",
        next_goal="search",
        action=[ActionModel(click=ClickAction(index=1))],
    )
    entry = StepHistoryEntry(
        step_number=0,
        agent_output=out,
        action_results=[ActionResult(action_name="click", success=True)],
        url_before="https://a.com",
        url_after="https://b.com",
        verification_result="[OK][VERIFIED] URL changed as predicted",
    )
    hist_text = entry.to_history_text()
    assert "Verification: [OK][VERIFIED] URL changed as predicted" in hist_text


def test_trace_logger(tmp_path: Path):
    run_dir = tmp_path / "test_run"
    logger = TraceLogger(run_dir=run_dir)
    
    pa = ProxyAssessment(
        target_element="input[1]",
        expected_state_delta="Text inserted",
        active_window_focus="Chrome",
        risk_label="safe",
    )
    out = AgentOutput(
        thinking="typing query",
        proxy_assessment=pa,
        next_goal="enter query",
        action=[ActionModel(click=ClickAction(index=1))],
    )
    v_res = VerificationResult(passed=True, label="VERIFIED", reason="Matched")
    
    logger.log_step(
        step_number=0,
        goal="Search papers",
        agent_output=out,
        action_results=[ActionResult(action_name="click", success=True)],
        verification_result=v_res,
        url_before="https://google.com",
        url_after="https://google.com/search?q=papers",
    )
    logger.finalize(success=True, result_text="Papers found")

    trace_file = run_dir / "traces.jsonl"
    assert trace_file.exists()
    
    lines = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    assert lines[0]["step"] == 0
    assert lines[0]["proxy_assessment"]["target_element"] == "input[1]"
    assert lines[0]["verification"]["label"] == "VERIFIED"
    assert lines[1]["type"] == "run_summary"
    assert lines[1]["success"] is True


def test_perception_manager_capture_diagnostic():
    import asyncio
    pm = PerceptionManager()
    snapshot = asyncio.run(pm.capture_diagnostic())
    assert isinstance(snapshot, PerceptionSnapshot)
    assert isinstance(snapshot.open_windows, list)
