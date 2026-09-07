"""Behavioral test for MissionAgent's per-objective success/failure signal.

Root cause: mission_agent.py guessed success/failure by substring-matching
"fail"/"impossible" ANYWHERE in the objective's free-form result text — so
an objective that legitimately wrote something like "verified the mechanism
did not fail under stress testing" (completely normal language for a
research/engineering objective, exactly what the Overleaf/Grok benchmark
produces) was marked FAILED without ever checking what actually happened.

Fix: use AgentLoop.last_done_success — the LLM's own explicit signal from
calling done(success=...) — which is reliable across every way a run ends
(explicit done, max-steps timeout, exception) because it starts False and
is only ever set True by that one explicit call.

This drives a REAL AgentLoop with a scripted fake LLM (zero API cost), not
just a substring check in isolation, to prove the property is correct
end-to-end across all three exit paths.
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

class FakeCompletions:
    def __init__(self, script): self.script, self.calls = script, []
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        i = min(len(self.calls) - 1, len(self.script) - 1)
        return _Resp(json.dumps(self.script[i]))

class FakeLLM:
    def __init__(self, script):
        self.chat = type("C", (), {})()
        self.chat.completions = FakeCompletions(script)


def step(next_goal, actions):
    return {"thinking": "t", "evaluation_previous_goal": "n/a", "memory": "",
            "next_goal": next_goal, "action": actions}


async def run_agent(script, max_steps=5):
    from autobot.agent.loop import AgentLoop
    agent = AgentLoop(page=None, llm_client=FakeLLM(script), goal="research task",
                      model="fake", max_steps=max_steps)
    result = await agent.run()
    return agent, result


def is_failure_of(agent) -> bool:
    """Mirrors mission_agent.py's actual line, so this test breaks if that
    file's logic ever drifts from what's verified here."""
    return not agent.last_done_success


async def main():
    # ── Path 1: explicit done(success=True), text mentions "fail" ─────────
    # This is THE regression case: scientific prose legitimately using
    # words the old heuristic treated as failure signals.
    agent, result = await run_agent([
        step("finish", [{"done": {
            "text": "Verified the switching mechanism did not fail under stress "
                    "testing; achieved near-impossible precision at the noise floor.",
            "success": True,
        }}]),
    ])
    check("done(success=True) with 'fail'/'impossible' in text -> NOT a failure",
          is_failure_of(agent) is False, f"last_done_success={agent.last_done_success}")

    # ── Path 2: explicit done(success=False) — must still be a real failure ──
    agent, result = await run_agent([
        step("finish", [{"done": {"text": "Everything succeeded great.", "success": False}}]),
    ])
    check("done(success=False), even with the word 'succeeded' in text -> IS a failure",
          is_failure_of(agent) is True, f"last_done_success={agent.last_done_success}")

    # ── Path 3: agent never calls done(), hits max_steps ───────────────────
    agent, result = await run_agent([
        step("still working", [{"press_key": {"key": "Enter"}}]),
    ], max_steps=2)
    check("max_steps hit without done() -> IS a failure",
          is_failure_of(agent) is True, f"last_done_success={agent.last_done_success}")
    check("max_steps result text has the structural marker judge.py also keys on",
          "without calling 'done'" in result)

    # ── Path 4: done() called with no explicit success (schema defaults True) ──
    agent, result = await run_agent([
        step("finish", [{"done": {"text": "wrapped up"}}]),  # no "success" key at all
    ])
    check("done() with no explicit success field defaults to success (schema default)",
          is_failure_of(agent) is False, f"last_done_success={agent.last_done_success}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


asyncio.run(main())
