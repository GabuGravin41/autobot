"""
Autobot 2.0 — End-to-End Perovskite Optical Memory & Overleaf Research Benchmark

Pipeline:
1. Open grok.com in user's real logged-in Chrome profile.
2. Conduct multi-step research on 'Perovskite cells for polariton exciton bistability cavities for optical memory'.
3. Ask follow-ups to refine paper, citations, and extract full LaTeX code.
4. Navigate to Overleaf.com in user's logged-in Chrome profile.
5. Click 'New Project' -> 'Blank Project'.
6. Name project 'Perovskite_Polariton_Exciton_Optical_Memory'.
7. Inject/Paste full LaTeX code into CodeMirror editor and trigger compilation (Ctrl+Enter).
"""
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("OverleafPerovskiteBenchmark")

load_dotenv()


def log_agent_progress(message: str) -> None:
    print(f"🤖 [AUTOBOT LIVE]: {message}")


async def run_benchmark():
    print("=" * 80)
    print("🧪 AUTOBOT 2.0 BENCHMARK: PEROVSKITE OPTICAL MEMORY & OVERLEAF AUTOMATION")
    print("=" * 80)

    from autobot.agent.runner import AgentRunner

    goal = """1. Navigate specifically to grok.com in Chrome.
2. Initiate a research conversation on Grok about: 'Using Perovskite cells to build polariton exciton bistability cavities for optical memory'.
3. Conduct a multi-turn conversation asking Grok to refine the academic paper structure, include mathematical models, and request the complete LaTeX source code with citations.
4. Extract the final LaTeX code for the paper.
5. Open a new tab and navigate to overleaf.com.
6. Click 'New Project' -> 'Blank Project'.
7. Name the project 'Perovskite_Polariton_Exciton_Optical_Memory' and click Create.
8. Inject/Paste the final LaTeX code into the Overleaf editor and trigger compilation (Ctrl+Enter).
9. Confirm PDF compilation success."""

    print(f"📋 BENCHMARK GOAL:\n{goal}\n")
    print("🚀 Initializing AgentRunner (Model: openai/gpt-4o)...")

    runner = AgentRunner.from_env(log_callback=log_agent_progress)
    
    try:
        result = await runner.run(goal=goal, max_steps=30)
        print("\n" + "=" * 80)
        print("🎉 BENCHMARK RUN COMPLETED!")
        print(f"Result Summary:\n{result[:1000]}")
        print("=" * 80)
    except Exception as e:
        logger.error(f"Benchmark run error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
