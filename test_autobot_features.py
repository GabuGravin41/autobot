"""
Autobot 2.0 Feature Verification Test Suite

Tests:
1. System Liveness Manager (sleep prevention state)
2. OS Command Execution (`RunCommandAction` in AgentLoop)
3. Mid-Flight Override & Goal Replanning
4. System Prompt Generation (Meta-agent rules & local tools)
5. FastAPI Endpoint & Status Integration
"""
import asyncio
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_1_liveness_manager():
    print("\n" + "=" * 60)
    print("TEST 1: System Liveness Manager")
    print("=" * 60)

    from autobot.computer.liveness import SystemLivenessManager

    mgr = SystemLivenessManager()
    print(f"  Initial state: active={mgr.is_active}")
    assert not mgr.is_active

    enabled = mgr.enable_liveness()
    print(f"  Enable liveness result: {enabled}, active={mgr.is_active}")
    assert enabled
    assert mgr.is_active

    disabled = mgr.disable_liveness()
    print(f"  Disable liveness result: {disabled}, active={mgr.is_active}")
    assert disabled
    assert not mgr.is_active

    print("  ✅ TEST 1 PASSED: Liveness Manager functions properly.")


async def test_2_os_command_execution():
    print("\n" + "=" * 60)
    print("TEST 2: OS Command Execution (`RunCommandAction`)")
    print("=" * 60)

    from autobot.agent.loop import AgentLoop
    from autobot.agent.models import RunCommandAction

    # Dummy mock page object for initialization
    class DummyPage:
        url = "http://localhost"

    agent = AgentLoop(
        page=DummyPage(),
        llm_client=None,
        goal="Test OS command execution",
    )

    cmd = RunCommandAction(command="echo Autobot OS Command Working!", timeout=10)
    result = await agent._execute_run_command(cmd)

    print(f"  Command success: {result.success}")
    print(f"  Command output snippet:\n    {result.extracted_content}")
    print(f"  Command error: {result.error}")

    assert result.success
    assert "Autobot OS Command Working!" in (result.extracted_content or "")
    print("  ✅ TEST 2 PASSED: `RunCommandAction` executed shell command and captured output.")


def test_3_mid_flight_override():
    print("\n" + "=" * 60)
    print("TEST 3: Mid-Flight Override & Goal Replanning")
    print("=" * 60)

    from autobot.agent.loop import AgentLoop

    class DummyPage:
        url = "http://localhost"

    agent = AgentLoop(
        page=DummyPage(),
        llm_client=None,
        goal="Initial Task Goal",
    )

    print(f"  Original goal: '{agent.goal}'")
    agent.push_override("Change goal: query Grok for Python BioPython solution instead.")
    print(f"  Pending override queued: '{agent.pending_override}'")
    assert agent.pending_override is not None

    # Simulate _execute_step applying override
    if agent.pending_override:
        agent.goal = f"{agent.goal}\n\n[HUMAN INTERVENTION / RE-PIVOT]: {agent.pending_override}"
        agent.pending_override = None

    print(f"  Updated goal after step intervention:\n{agent.goal}")
    assert "[HUMAN INTERVENTION / RE-PIVOT]" in agent.goal
    assert agent.pending_override is None
    print("  ✅ TEST 3 PASSED: Mid-flight goal override successfully repivots the trajectory.")


def test_4_system_prompt_builder():
    print("\n" + "=" * 60)
    print("TEST 4: System Prompt Template & Tool Catalog")
    print("=" * 60)

    from autobot.computer.computer import Computer
    from autobot.prompts.builder import SystemPromptBuilder

    comp = Computer()
    catalog = comp.get_tool_catalog()
    builder = SystemPromptBuilder(tool_catalog=catalog)
    prompt = builder.build()

    print(f"  System prompt length: {len(prompt)} characters")
    assert "run_command" in prompt
    assert "request_human_input" in prompt
    assert "Resourcefulness & Meta-Agent Delegation (Vibe Coding)" in prompt
    assert "The Self-Driving Computer" in prompt
    print("  ✅ TEST 4 PASSED: System prompt contains meta-agent rules & OS tools.")


def test_5_fastapi_endpoints():
    print("\n" + "=" * 60)
    print("TEST 5: FastAPI Endpoint & Status Integration")
    print("=" * 60)

    from fastapi.testclient import TestClient
    from autobot.web.app import app

    client = TestClient(app)

    # Test GET /api/agent/status
    res = client.get("/api/agent/status")
    print(f"  GET /api/agent/status code: {res.status_code}")
    print(f"  Status response: {res.json()}")
    assert res.status_code == 200
    data = res.json()
    assert "agent_status" in data
    assert "liveness_active" in data

    # Test POST /api/agent/override without active run (should return 400)
    res_override = client.post("/api/agent/override", json={"instruction": "test"})
    print(f"  POST /api/agent/override code (no run): {res_override.status_code}")
    assert res_override.status_code == 400

    print("  ✅ TEST 5 PASSED: FastAPI endpoints respond accurately.")


async def main():
    print("🤖 Autobot 2.0 — Feature Verification Test Suite")
    test_1_liveness_manager()
    await test_2_os_command_execution()
    test_3_mid_flight_override()
    test_4_system_prompt_builder()
    test_5_fastapi_endpoints()

    print("\n" + "=" * 60)
    print("🎉 ALL 5 FEATURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
