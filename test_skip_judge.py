"""Behavioral test for AUTOBOT_SKIP_JUDGE.

Verifies AgentRunner actually skips the Judge Agent's extra LLM call when
the flag is set, uses the run's own done(success=...) signal instead, and
that the LLM client's call count proves no second call happened - not just
that the code reads like it should.
"""
import asyncio
import json
import os
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
    """Records every call, tagged by which model/purpose requested it, so we
    can tell an agent-loop call apart from a judge call."""
    def __init__(self, script):
        self.script = script
        self.calls = []

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


async def run_with_flag(skip: bool):
    from autobot.agent.runner import AgentRunner

    # Single script entry: the agent immediately calls done(success=True).
    # If Judge is NOT skipped, it makes a SECOND call using the exact same
    # fake client — the script only has one entry, so a second call would
    # reuse index 0 (agent's own JSON), which Judge would then fail to
    # parse as {"success":..., "reasoning":...} and report success=False.
    # That mismatch is exactly how we detect whether a second call happened.
    llm = CountingLLM([step("finish", [{"done": {"text": "did it", "success": True}}])])

    if skip:
        os.environ["AUTOBOT_SKIP_JUDGE"] = "1"
    else:
        os.environ.pop("AUTOBOT_SKIP_JUDGE", None)

    try:
        runner = AgentRunner(llm_client=llm, model="fake", max_steps=3)
        result = await runner.run("a simple task with no browser")
        return result, len(llm.chat.completions.calls)
    finally:
        os.environ.pop("AUTOBOT_SKIP_JUDGE", None)


async def main():
    result_skip, calls_skip = await run_with_flag(skip=True)
    check("AUTOBOT_SKIP_JUDGE=1 makes exactly ONE LLM call (agent only, no judge)",
          calls_skip == 1, f"calls={calls_skip}")
    check("result reflects success via the run's own done(success=True) signal",
          "SUCCESS" in result_skip or "success" in result_skip.lower(), result_skip[:200])
    check("skipped-judge reasoning says so explicitly (no false claim of LLM verification)",
          "skipped" in result_skip.lower() or "AUTOBOT_SKIP_JUDGE" in result_skip, result_skip[:300])

    result_full, calls_full = await run_with_flag(skip=False)
    check("without the flag, a SECOND call happens (the real Judge behavior)",
          calls_full == 2, f"calls={calls_full}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


asyncio.run(main())
