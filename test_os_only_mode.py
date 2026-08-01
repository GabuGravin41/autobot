"""End-to-end AgentLoop test with a scripted fake LLM - costs nothing to run.

This is the harness that was missing. Every serious bug in this project lived
in code that compiled fine and had simply never been executed; the only way to
catch that class of bug cheaply is to actually drive the loop. A scripted LLM
lets us do that with zero API spend and zero nondeterminism.

Specifically covers the failure seen live: "open Notepad and type hello" died
because AgentRunner unconditionally required a CDP-attached Chrome, so a task
needing no browser at all never reached step 1.

Actions used here are READ-ONLY (window.list_all) so running this never moves
the mouse, types, or opens anything.

Run:  python test_os_only_mode.py
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


# ── A fake OpenAI-compatible client that replays a scripted list of steps ──
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

class FakeChat:
    def __init__(self, script): self.completions = FakeCompletions(script)

class FakeLLM:
    """Mimics the openai client surface AgentLoop actually uses."""
    def __init__(self, script): self.chat = FakeChat(script)


def step(next_goal, actions, thinking="test"):
    return {
        "thinking": thinking,
        "evaluation_previous_goal": "n/a",
        "memory": "",
        "next_goal": next_goal,
        "action": actions,
    }


async def main():
    from autobot.agent.loop import AgentLoop

    # ── 1. AgentLoop must CONSTRUCT with no browser attached ──────────────
    try:
        script = [
            step("List open windows", [{"computer_call": {"call": "computer.window.list_all()"}}]),
            step("Finish", [{"done": {"text": "listed windows", "success": True}}]),
        ]
        llm = FakeLLM(script)
        agent = AgentLoop(page=None, llm_client=llm, goal="list the open windows",
                          model="fake", max_steps=4)
        check("AgentLoop constructs with page=None", True)
    except Exception as e:
        check("AgentLoop constructs with page=None", False, repr(e))
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
        sys.exit(1)

    # ── 2. It must RUN a full step cycle with no browser ──────────────────
    try:
        result = await agent.run()
        check("run() completes without a browser", isinstance(result, str), repr(result))
        check("reached the done action", "listed windows" in (result or ""), repr(result))
    except Exception as e:
        import traceback; traceback.print_exc()
        check("run() completes without a browser", False, repr(e))
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
        sys.exit(1)

    # ── 3. The computer_call actually dispatched to the real Computer ─────
    first = agent.history[0].action_results[0] if agent.history else None
    check("computer_call executed", first is not None and first.action_name == "computer_call",
          repr(first))
    check("computer_call succeeded", first is not None and first.success,
          getattr(first, "error", None))
    check("returned real window data",
          first is not None and bool(first.extracted_content),
          repr(getattr(first, "extracted_content", None))[:200])

    # ── 4. Browser-only actions must fail LOUDLY and usefully, not crash ──
    agent2 = AgentLoop(page=None, llm_client=FakeLLM([step("nav", [])]), goal="x",
                       model="fake", max_steps=1)
    from autobot.agent.models import ActionModel
    nav = ActionModel(navigate={"url": "https://example.com"})
    res = await agent2._execute_single_action(nav, _empty_state())
    check("navigate without browser fails cleanly", not res.success, repr(res))
    check("error explains the OS-only alternative",
          "computer_call" in (res.error or ""), repr(res.error))

    # ── 5. _page_url() is null-safe (this used to raise AttributeError) ───
    check("_page_url() safe with no page", agent2._page_url() == "")

    # ── 6. Vision is skipped when there's nothing to look at -------------
    check("vision decision runs without a browser",
          isinstance(agent2._should_use_vision(_empty_state()), bool))

    # ── 7. An unreachable LLM must fail FAST, not burn the step budget ───
    # Regression: with the LLM down, the agent spun through all 25 steps and
    # then reported "No steps were executed", telling the user nothing.
    from autobot.agent.loop import LLMUnavailableError

    class DeadCompletions:
        def __init__(self): self.attempts = 0
        async def create(self, **kwargs):
            self.attempts += 1
            raise ConnectionError("simulated: API unreachable")
    class DeadLLM:
        def __init__(self):
            self.chat = type("C", (), {})()
            self.chat.completions = DeadCompletions()

    dead = DeadLLM()
    agent3 = AgentLoop(page=None, llm_client=dead, goal="x", model="fake", max_steps=25)
    try:
        await agent3.run()
        check("unreachable LLM raises instead of spinning", False, "run() returned normally")
    except LLMUnavailableError as e:
        check("unreachable LLM raises instead of spinning", True)
        check("aborts after 3 attempts, not 25",
              dead.chat.completions.attempts == 3, f"attempts={dead.chat.completions.attempts}")
        check("error names the underlying cause",
              "unreachable" in str(e) or "ConnectionError" in str(e), str(e)[:120])
    except Exception as e:
        check("unreachable LLM raises instead of spinning", False, f"wrong type: {type(e).__name__}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


def _empty_state():
    from autobot.dom.models import BrowserState, DOMElementNode, SelectorMap
    return BrowserState(
        url="", title="", tabs=[], page_info=None,
        element_tree=DOMElementNode(index=None, tag_name="body", text="",
                                    attributes={}, children=[], depth=0),
        selector_map=SelectorMap(), screenshot_b64=None,
    )


asyncio.run(main())
