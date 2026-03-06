"""
Autobot New Architecture — End-to-End Test Script

Runs layered tests:
    Layer 1: Import verification
    Layer 2: CDP browser connection
    Layer 3: DOM extraction from a real page
    Layer 4: Full agent loop (1 step, real LLM call)
"""
import asyncio
import os
import sys
import traceback

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()


async def test_layer_1_imports():
    """Test all new module imports."""
    print("\n" + "="*60)
    print("LAYER 1: Import Verification")
    print("="*60)

    modules = [
        ("autobot.dom.models", "DOMElementNode, SelectorMap, BrowserState"),
        ("autobot.dom.extraction", "DOMExtractionService"),
        ("autobot.prompts.builder", "SystemPromptBuilder, StepPromptBuilder"),
        ("autobot.agent.models", "AgentOutput, ActionModel, ActionResult"),
        ("autobot.agent.loop", "AgentLoop"),
        ("autobot.agent.runner", "AgentRunner"),
        ("autobot.computer.computer", "Computer"),
        ("autobot.browser.launcher", "AsyncBrowserLauncher"),
    ]

    all_ok = True
    for module_path, classes in modules:
        try:
            mod = __import__(module_path, fromlist=classes.split(", "))
            for cls_name in classes.split(", "):
                getattr(mod, cls_name.strip())
            print(f"  ✅ {module_path}: {classes}")
        except Exception as e:
            print(f"  ❌ {module_path}: {e}")
            all_ok = False

    # Test tool catalog generation
    from autobot.computer.computer import Computer
    c = Computer()
    catalog = c.get_tool_catalog()
    tool_count = catalog.count("computer.")
    print(f"  ✅ Tool catalog: {tool_count} tools auto-generated")

    # Test system prompt loading
    from autobot.prompts.builder import SystemPromptBuilder
    sp = SystemPromptBuilder()
    prompt = sp.build()
    print(f"  ✅ System prompt: {len(prompt)} chars loaded")

    return all_ok


async def test_layer_2_cdp_connect():
    """Test CDP browser connection."""
    print("\n" + "="*60)
    print("LAYER 2: CDP Browser Connection")
    print("="*60)

    from autobot.browser.launcher import AsyncBrowserLauncher

    launcher = AsyncBrowserLauncher.from_env()
    print(f"  Chrome path: {launcher.chrome_path}")
    print(f"  User data dir: {launcher.user_data_dir}")
    print(f"  CDP port: {launcher.debug_port}")

    try:
        page = await launcher.start()
        print(f"  ✅ Connected! Current URL: {page.url}")

        # Navigate to a test page
        await page.goto("https://www.google.com", wait_until="domcontentloaded")
        print(f"  ✅ Navigated to: {page.url}")
        title = await page.title()
        print(f"  ✅ Page title: {title}")

        return launcher, page
    except Exception as e:
        print(f"  ❌ CDP connection failed: {e}")
        traceback.print_exc()
        return None, None


async def test_layer_3_dom_extraction(page):
    """Test DOM extraction from a real page."""
    print("\n" + "="*60)
    print("LAYER 3: DOM Extraction")
    print("="*60)

    if page is None:
        print("  ⏭️ Skipped (no browser connection)")
        return False

    from autobot.dom.extraction import DOMExtractionService

    try:
        dom_service = DOMExtractionService(page)
        browser_state = await dom_service.extract_state()

        print(f"  ✅ URL: {browser_state.url}")
        print(f"  ✅ Title: {browser_state.title}")
        print(f"  ✅ Interactive elements: {browser_state.num_interactive}")
        print(f"  ✅ Total elements: {browser_state.total_elements}")
        print(f"  ✅ Links: {browser_state.num_links}")
        print(f"  ✅ Selector map entries: {len(browser_state.selector_map)}")

        if browser_state.screenshot_b64:
            print(f"  ✅ Screenshot captured ({len(browser_state.screenshot_b64)} chars b64)")
        else:
            print(f"  ⚠️ No screenshot (not critical)")

        # Show first few interactive elements
        from autobot.dom.models import DOMSerializedState
        serialized = DOMSerializedState(
            element_tree=browser_state.element_tree,
            selector_map=browser_state.selector_map,
        )
        llm_text = serialized.llm_representation()
        lines = llm_text.strip().split("\n")[:10]
        print(f"\n  DOM Tree (first 10 lines):")
        for line in lines:
            print(f"    {line}")

        return True
    except Exception as e:
        print(f"  ❌ DOM extraction failed: {e}")
        traceback.print_exc()
        return False


async def test_layer_4_agent_step(page):
    """Test one step of the agent loop (real LLM call)."""
    print("\n" + "="*60)
    print("LAYER 4: Agent Loop (1 step)")
    print("="*60)

    if page is None:
        print("  ⏭️ Skipped (no browser connection)")
        return False

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⏭️ Skipped (no LLM API key)")
        return False

    from autobot.agent.runner import AgentRunner, _create_llm_client
    from autobot.browser.launcher import AsyncBrowserLauncher

    try:
        client = _create_llm_client()
        model = os.getenv("AUTOBOT_LLM_MODEL", "gpt-4o")
        print(f"  LLM: {model}")
        print(f"  Goal: 'You are on Google. Type hello world in the search box.'")
        print(f"  Max steps: 3 (just testing)")

        from autobot.agent.loop import AgentLoop
        agent = AgentLoop(
            page=page,
            llm_client=client,
            goal="You are on Google. Type 'hello world' in the search box and press Enter.",
            model=model,
            max_steps=3,
            use_vision=False,  # Save tokens on first test
        )

        result = await agent.run()
        print(f"\n  ✅ Agent finished!")
        print(f"  Result: {result[:200]}")
        print(f"  Steps executed: {len(agent.history)}")
        for entry in agent.history:
            print(f"    Step {entry.step_number + 1}: {entry.agent_output.next_goal[:60]}")
            for ar in entry.action_results:
                icon = "✅" if ar.success else "❌"
                print(f"      {icon} {ar.action_name}")

        return True
    except Exception as e:
        print(f"  ❌ Agent loop failed: {e}")
        traceback.print_exc()
        return False


async def main():
    print("🤖 Autobot New Architecture — End-to-End Tests")
    print("=" * 60)

    results = {}

    # Layer 1: Imports
    results["imports"] = await test_layer_1_imports()

    # Layer 2: CDP
    launcher, page = await test_layer_2_cdp_connect()
    results["cdp"] = page is not None

    # Layer 3: DOM
    results["dom"] = await test_layer_3_dom_extraction(page)

    # Layer 4: Agent (real LLM call)
    results["agent"] = await test_layer_4_agent_step(page)

    # Cleanup
    if launcher:
        await launcher.stop()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")

    all_passed = all(results.values())
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '⚠️ Some tests failed — see above.'}")
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
