"""
Live Agent Loop End-to-End Test using OpenRouter GPT-4o
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

async def test_live_step():
    print("🤖 Testing Live Agent Loop with GPT-4o via OpenRouter...")
    from autobot.agent.runner import AgentRunner, _create_llm_client
    from autobot.agent.loop import AgentLoop
    from autobot.dom.extraction import DOMExtractionService

    class MockPage:
        url = "https://www.google.com"

        async def title(self):
            return "Google"

        def is_closed(self):
            return False

        async def evaluate(self, script):
            return {
                "scrollY": 0, "scrollX": 0,
                "viewportHeight": 1080, "viewportWidth": 1920,
                "pageHeight": 1080, "pageWidth": 1920
            }

        class Accessibility:
            async def snapshot(self):
                return {
                    "role": "WebArea",
                    "name": "Google",
                    "children": [
                        {"role": "searchbox", "name": "Search", "value": ""},
                        {"role": "button", "name": "Google Search"},
                    ]
                }

        accessibility = Accessibility()

        async def screenshot(self, type="png"):
            return b"fake_png_data"

        class Context:
            pages = []
        context = Context()

    client = _create_llm_client()
    model = os.getenv("AUTOBOT_LLM_MODEL", "openai/gpt-4o")
    print(f"  Model: {model}")
    print(f"  Client created: {client is not None}")

    agent = AgentLoop(
        page=MockPage(),
        llm_client=client,
        goal="You are on Google. Search for VESTA materials science software download link.",
        model=model,
        max_steps=2,
        use_vision=False,
    )

    print("🧠 Calling GPT-4o for Agent Thinking & Action generation...")
    dom_service = DOMExtractionService(agent.page)
    browser_state = await dom_service.extract_state()

    output = await agent._call_llm(browser_state)
    if output:
        print("\n✅ LIVE AGENT LLM RESPONSE RECEIVED FROM GPT-4o!")
        print(f"  Thinking: {output.thinking}")
        print(f"  Next Goal: {output.next_goal}")
        print(f"  Action Count: {len(output.action)}")
        for a in output.action:
            print(f"    Action: {a.action_name} -> {a.action_data}")
        return True
    else:
        print("❌ LLM output was None")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_live_step())
    if success:
        print("\n🎉 LIVE E2E AGENT STEP TEST PASSED!")
    else:
        print("\n⚠️ Live agent step test failed.")
