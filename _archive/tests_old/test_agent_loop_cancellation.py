"""Behavioral test for AgentLoop cancellation, including the pre-start race.

Root cause: run() used to reset `self.is_cancelled = False` on entry. But
AgentRunner.cancel() can reach an AgentLoop instance and set
is_cancelled=True in the real window between construction and `await
agent.run()` actually starting (runner.py's _run_single_loop/_run_mission
build the AgentLoop, then await it a few lines later — a dashboard "cancel"
click landing in that window is entirely plausible under any latency). The
reset on entry would silently discard that cancel request, running the
full task anyway with no visible sign the cancellation had been lost.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


class _Msg:
    def __init__(self, content): self.content = content
class _Choice:
    def __init__(self, content): self.message = _Msg(content)
class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]

class CountingCompletions:
    def __init__(self, script):
        self.script, self.calls = script, []
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        i = min(len(self.calls) - 1, len(self.script) - 1)
        return _Resp(json.dumps(self.script[i]))

class CountingLLM:
    def __init__(self, script):
        self.chat = type("C", (), {})()
        self.chat.completions = CountingCompletions(script)


def step(next_goal, actions):
    return {"thinking": "t", "evaluation_previous_goal": "n/a", "memory": "",
            "next_goal": next_goal, "action": actions}


async def main():
    from autobot.agent.loop import AgentLoop

    # ── The regression: cancel() lands BEFORE run() is ever awaited ───────
    llm = CountingLLM([step("finish", [{"done": {"text": "should never get here", "success": True}}])])
    agent = AgentLoop(page=None, llm_client=llm, goal="x", model="fake", max_steps=5)

    # Simulates AgentRunner.cancel() reaching this instance in the window
    # between construction and the eventual `await agent.run()`.
    agent.is_cancelled = True

    result = await agent.run()
    check("cancel-before-start is honored, not silently reset",
          result == "Task cancelled by user.", repr(result))
    check("no LLM call was made — cancelled before doing any real work",
          len(llm.chat.completions.calls) == 0, f"calls={len(llm.chat.completions.calls)}")

    # ── Normal case: no cancellation, run completes as expected ────────────
    llm2 = CountingLLM([step("finish", [{"done": {"text": "done normally", "success": True}}])])
    agent2 = AgentLoop(page=None, llm_client=llm2, goal="x", model="fake", max_steps=5)
    result2 = await agent2.run()
    check("uncancelled run still completes normally (no regression)",
          result2 == "done normally", repr(result2))
    check("uncancelled run actually called the LLM",
          len(llm2.chat.completions.calls) == 1)

    # ── Mid-run cancellation: set after step 1 starts, must halt before step 2 ──
    llm3 = CountingLLM([
        step("step one", [{"press_key": {"key": "Enter"}}]),
        step("step two — should never run", [{"done": {"text": "too late", "success": True}}]),
    ])
    agent3 = AgentLoop(page=None, llm_client=llm3, goal="x", model="fake", max_steps=5)

    original_execute_step = agent3._execute_step
    async def _cancel_after_first_step():
        result = await original_execute_step()
        agent3.is_cancelled = True  # simulate cancel() arriving mid-run
        return result
    agent3._execute_step = _cancel_after_first_step

    result3 = await agent3.run()
    check("mid-run cancellation halts before the next step",
          result3 == "Task cancelled by user.", repr(result3))
    check("only the first step's LLM call happened, not the second",
          len(llm3.chat.completions.calls) == 1, f"calls={len(llm3.chat.completions.calls)}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


asyncio.run(main())
