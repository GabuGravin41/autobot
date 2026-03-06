import asyncio
import logging
from autobot.agent.runner import AgentRunner
import os

# Set up logging to show agent progress
logging.basicConfig(level=logging.INFO, format="%(message)s")

async def test_judge():
    os.environ["AUTOBOT_BROWSER_MODE"] = "cdp"
    # Create the runner
    runner = AgentRunner.from_env()
    
    # We will ask it to do something very simple to see if the judge catches it.
    goal = "Go to https://news.ycombinator.com/ and search for 'Agentic AI', then tell me the top result"
    
    print(f"Goal: {goal}")
    print("Running...")
    
    try:
        result = await runner.run(goal, max_steps=5)
        print("\n\n=== FINAL RESULT ===")
        print(result)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_judge())
