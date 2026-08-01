"""
Interactive Single-Step Debugger & Verbose Execution Loop

Runs the agent step-by-step with real-time screen inspection, verbose DOM logs,
and instant pause-on-error behavior to debug navigation & Overleaf automation.
"""
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Force UTF-8 encoding on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure verbose debug logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DebugLoop")

load_dotenv()


async def debug_agent_run():
    print("=" * 80)
    print("🔍 AUTOBOT 2.0 — INTERACTIVE VERBOSE STEP-BY-STEP DEBUGGER")
    print("=" * 80)

    from autobot.browser.launcher import AsyncBrowserLauncher
    from autobot.agent.runner import _create_llm_client
    from autobot.agent.loop import AgentLoop
    from autobot.dom.extraction import DOMExtractionService

    # Step 1: Connect strictly to real Chrome profile
    print("\n🌐 Step 1: Connecting to Real Chrome Profile...")
    launcher = AsyncBrowserLauncher.from_env()
    page = await launcher.start()

    print(f"  ✅ Page Connected: '{page.url}'")

    # Step 2: Initialize LLM Client
    client = _create_llm_client()
    model = os.getenv("AUTOBOT_LLM_MODEL", "openai/gpt-4o")
    print(f"  🤖 LLM Model: '{model}' | Client Active: {client is not None}")

    goal = "Navigate to grok.com and verify the user is logged in to Grok."

    agent = AgentLoop(
        page=page,
        llm_client=client,
        goal=goal,
        model=model,
        max_steps=5,
        use_vision=True,
    )

    print(f"\n📋 Task Goal: '{goal}'")
    print("-" * 80)

    # Step 3: Run single steps with verbose inspection
    for step in range(1, 6):
        print(f"\n▶️ EXECUTING DEBUG STEP {step}/5...")
        
        # Inspect current page state
        dom_service = DOMExtractionService(page)
        b_state = await dom_service.extract_state()

        elem_count = len(b_state.selector_map) if b_state.selector_map else 0
        print(f"  📍 Current Page URL: {b_state.url}")
        print(f"  📄 Page Title: '{b_state.title}'")
        print(f"  🔍 Interactive Elements Found (selector_map): {elem_count}")
        print(f"  📸 Screenshot Size: {len(b_state.screenshot_b64 or '')} bytes")

        # Execute 1 step
        print("  🧠 Querying LLM for step decision...")
        action_summary = await agent._execute_step()

        print(f"  📝 Agent Output: {action_summary}")

        if agent.history:
            last_entry = agent.history[-1]
            print(f"  💭 Thinking: {last_entry.agent_output.thinking}")
            print(f"  🎯 Next Goal: {last_entry.agent_output.next_goal}")
            for ar in last_entry.action_results:
                icon = "✅" if ar.success else "❌"
                print(f"    {icon} Action '{ar.action_name}' -> Success: {ar.success}")
                if ar.error:
                    print(f"       ⚠️ Error: {ar.error}")
                    print(f"⛔ STOPPING DEBUG RUN ON STEP {step} ERROR FOR INSTANT FIX.")
                    return

        if action_summary and "Task completed" in action_summary:
            print("\n🎉 TASK COMPLETED SUCCESSFULLY IN DEBUG LOOP!")
            break

        await asyncio.sleep(2.0)

    await launcher.stop()


if __name__ == "__main__":
    asyncio.run(debug_agent_run())
