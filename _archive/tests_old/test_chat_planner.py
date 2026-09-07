"""Behavioral test for POST /api/chat (the "AI Planner").

Root cause of the reported bug: this endpoint never called an LLM at all —
it synchronously fabricated an identical single-step "Auto Task" plan from
whatever text was typed, regardless of content, and silently dropped the
conversation history the frontend was already sending (ChatRequest had no
history field). "An execution plan appears immediately, for no clear
reason" is exactly what a stub with zero await points looks like from the
outside.

This drives the real chat() function with a scripted fake LLM (zero API
cost), proving: different input produces different output (not a stub),
multi-turn history actually reaches the model, a no-plan/clarifying-
question turn returns plan=None, malformed LLM JSON degrades gracefully
instead of crashing, and no-API-key degrades gracefully too.
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

class ScriptedCompletions:
    def __init__(self, script):
        self.script, self.calls = script, []
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        i = min(len(self.calls) - 1, len(self.script) - 1)
        content = self.script[i]
        return _Resp(content if isinstance(content, str) else json.dumps(content))

class ScriptedLLM:
    def __init__(self, script):
        self.chat = type("C", (), {})()
        self.chat.completions = ScriptedCompletions(script)


async def call_chat(message, history=None, script=None, llm_client=None):
    """Drives the real chat() handler with a patched _create_llm_client."""
    from autobot.web import app as app_module

    fake_client = llm_client if llm_client is not None else (ScriptedLLM(script) if script else None)

    import autobot.agent.runner as runner_module
    original = runner_module._create_llm_client
    runner_module._create_llm_client = lambda: fake_client
    try:
        req = app_module.ChatRequest(message=message, history=history or [])
        result = await app_module.chat(req)
        return result, fake_client
    finally:
        runner_module._create_llm_client = original


async def main():
    # ── THE regression: different input -> different output ────────────────
    result_a, _ = await call_chat("open notepad and type hello", script=[
        {"reply": "I'll open Notepad and type hello.", "needs_plan": True,
         "plan_name": "Notepad Test", "plan_description": "Open Notepad and type hello",
         "plan_steps": ["Focus or open Notepad", "Type 'hello'"]},
    ])
    result_b, _ = await call_chat("research perovskite polaritons and write a paper", script=[
        {"reply": "I'll research this and draft a paper.", "needs_plan": True,
         "plan_name": "Research Paper", "plan_description": "Research and write about perovskite polaritons",
         "plan_steps": ["Search Grok for background", "Draft LaTeX", "Compile in Overleaf"]},
    ])
    check("different messages produce genuinely different plans (not a stub)",
          result_a["plan"]["name"] != result_b["plan"]["name"]
          and result_a["plan"]["description"] != result_b["plan"]["description"],
          f"a={result_a['plan']} b={result_b['plan']}")
    check("plan shape matches the frontend's BackendPlan contract",
          all(k in result_a["plan"] for k in ("id", "name", "description", "steps"))
          and all("description" in s and "action" in s for s in result_a["plan"]["steps"]))

    # ── Multi-turn history actually reaches the model ───────────────────────
    _, client = await call_chat(
        "yes, do that",
        history=[{"role": "user", "content": "open overleaf"},
                  {"role": "assistant", "content": "Should I create a new project?"}],
        script=[{"reply": "Creating it now.", "needs_plan": True, "plan_name": "Overleaf",
                 "plan_description": "Create an Overleaf project", "plan_steps": ["Create project"]}],
    )
    sent_messages = client.chat.completions.calls[0]["messages"]
    check("history is actually included in the LLM call",
          any(m.get("content") == "open overleaf" for m in sent_messages)
          and any(m.get("content") == "Should I create a new project?" for m in sent_messages),
          str(sent_messages))
    check("the new user message is the last one sent",
          sent_messages[-1] == {"role": "user", "content": "yes, do that"})

    # ── Clarifying question: needs_plan=False -> plan is None ──────────────
    result_q, _ = await call_chat("automate my computer", script=[
        {"reply": "Sure — what would you like automated specifically?",
         "needs_plan": False, "plan_steps": []},
    ])
    check("a clarifying-question turn returns plan=None, not a fabricated plan",
          result_q["plan"] is None, str(result_q))
    check("the clarifying question itself is passed through as the reply",
          "specifically" in result_q["reply"])

    # ── Malformed LLM output degrades gracefully, doesn't crash ─────────────
    result_bad, _ = await call_chat("test", script=["not valid json at all, just prose"])
    check("malformed JSON doesn't crash the endpoint", result_bad is not None)
    check("malformed JSON falls back to a plain-text reply", result_bad["plan"] is None)

    # ── No LLM configured -> honest message, not a crash or a fake plan ────
    result_none, _ = await call_chat("test", llm_client=None)
    check("no LLM configured returns an honest explanation, not a fabricated plan",
          result_none["plan"] is None and "doctor" in result_none["reply"].lower())

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


asyncio.run(main())
