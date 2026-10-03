"""
LLM backend layer + the weak-model regressions found in the Sep 2026 review.

Regressions covered (all reproduced against the old code first):
  - a malformed action ended the run as "Task complete" with success=True
  - failed runs were saved as "proven" skills and re-injected later
  - the model only saw the previous step, so it looped (Sep 7 Overleaf run)
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autobot.agent.action_models import Action, LLMResponse, StepRecord
from autobot.llm.backends import (
    ChainBackend,
    ChatCompletionsShim,
    ClaudeCLIBackend,
    LLMBackend,
    LLMError,
    OpenAICompatBackend,
)
from autobot.util.jsonx import coerce_call, extract_json

KNOWN = {"click": ["index"], "type": ["text"], "run_shell": ["command", "timeout"], "done": ["text", "success"]}


# ── tolerant parsing ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    '{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go: {"a": 1} hope that helps',
    "{'a': 1}", '{"a": 1,}', 'noise {"a": 1} {"b": 2}',
])
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_nothing():
    assert extract_json("no json here") is None


@pytest.mark.parametrize("data,expected", [
    ({"name": "click", "params": {"index": 3}}, ("click", {"index": 3})),
    ({"name": "click", "args": {"index": 3}}, ("click", {"index": 3})),
    ({"type": "click", "index": 3}, ("click", {"index": 3})),                 # flat args
    ({"tool": "run_shell", "args": ["dir", 10]}, ("run_shell", {"command": "dir", "timeout": 10})),
    ({"click": {"index": 3}}, ("click", {"index": 3})),
    ({"function": {"name": "click", "arguments": '{"index": 4}'}}, ("click", {"index": 4})),
    ("click(3)", ("click", {"index": 3})),
    ("call: type(text='hi')", ("type", {"text": "hi"})),
    ({"type": "hello"}, ("type", {"text": "hello"})),
])
def test_coerce_call_shapes(data, expected):
    assert coerce_call(data, KNOWN) == expected


@pytest.mark.parametrize("data", [{"name": "explode"}, {"foo": 1, "bar": 2}, "os.system('x')", [], None])
def test_coerce_call_never_guesses(data):
    assert coerce_call(data, KNOWN) is None


def test_malformed_action_is_not_done():
    # Old behavior: this became done(success=True) and ended the run.
    r = LLMResponse.from_dict({"thinking": "", "next_goal": "click OK", "action": {"type": "click", "index": 1}})
    assert r.action.name == "click" and r.action.params == {"index": 1}
    assert LLMResponse.from_dict({"thinking": "", "action": "gibberish"}) is None


def test_done_needs_explicit_success():
    r = LLMResponse.from_dict({"action": {"name": "done", "params": {"text": "finished"}}})
    assert r.action.params["success"] is False
    r = LLMResponse.from_dict({"action": {"name": "done", "params": {"text": "ok", "success": "true"}}})
    assert r.action.params["success"] is True


# ── CoreLoop with a weak model ───────────────────────────────────────────────

class _Window:
    def __init__(self):
        self.clicks = 0

    def extract_ui(self):
        return "[1] Button 'OK'"

    def active_title(self):
        return "Notepad"

    def click(self, i):
        self.clicks += 1
        return True


class _Computer:
    def __init__(self):
        self.window = _Window()

    def get_tool_catalog(self):
        return ""


def _llm(replies):
    replies = list(replies)
    seen = []

    def create(**kw):
        seen.append(kw["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), seen


def test_coreloop_malformed_then_valid(monkeypatch):
    from autobot.agent.core_loop import CoreLoop
    llm, seen = _llm([
        json.dumps({"thinking": "t", "next_goal": "n", "action": "click the ok button please"}),  # unusable
        json.dumps({"thinking": "t", "next_goal": "n", "action": {"type": "click", "index": 1}}),  # flat args
        json.dumps({"thinking": "t", "next_goal": "n", "action": {"name": "done", "params": {"text": "ok", "success": True}}}),
    ])
    comp = _Computer()
    loop = CoreLoop(computer=comp, llm_client=llm, goal="press OK", max_steps=5)
    result = asyncio.run(loop.run())
    assert comp.window.clicks == 1
    assert loop._last_done_success is True and result == "ok"
    assert "Valid action names" in seen[1][-1]["content"]      # the re-ask told it what's valid


def test_coreloop_unparseable_forever_does_not_claim_success():
    from autobot.agent.core_loop import CoreLoop
    llm, _ = _llm(["nonsense"] * 10)
    loop = CoreLoop(computer=_Computer(), llm_client=llm, goal="g", max_steps=3)
    result = asyncio.run(loop.run())
    assert loop._last_done_success is False
    assert "Task complete" not in result


def test_coreloop_blocks_identical_repeats_and_shows_history():
    from autobot.agent.core_loop import CoreLoop
    act = json.dumps({"thinking": "t", "next_goal": "n", "action": {"name": "computer_call", "params": {"call": "computer.x()"}}})
    done = json.dumps({"thinking": "t", "next_goal": "n", "action": {"name": "done", "params": {"text": "gave up", "success": False}}})
    llm, seen = _llm([act] * 5 + [done])
    loop = CoreLoop(computer=_Computer(), llm_client=llm, goal="g", max_steps=10)
    loop._dispatch = lambda action: "same result"
    asyncio.run(loop.run())
    results = [h.result for h in loop.history[:-1]]
    assert results[:3] == ["same result"] * 3
    assert all(r.startswith("BLOCKED") for r in results[3:])
    last_prompt = seen[-1][-1]["content"]
    assert "# Steps So Far" in last_prompt and "Already Tried" in last_prompt


def test_failed_runs_are_not_distilled(tmp_path):
    from autobot.knowledge.skill_distiller import SkillDistiller
    d = SkillDistiller(skills_dir=tmp_path)
    hist = [StepRecord(step=i, next_goal="g", action=Action("computer_call", {"call": "computer.browser.url()"}),
                       result="x", success=True, screen_before="", screen_after="") for i in range(12)]
    assert d.distill_from_run(goal="Open Overleaf", history=hist, result="failed") is None
    assert list(tmp_path.glob("*.json")) == []
    assert d.distill_from_run(goal="Open Overleaf", history=hist, result="success") is not None


# ── backends ─────────────────────────────────────────────────────────────────

class _Fake(LLMBackend):
    def __init__(self, name, out=None, err=None):
        self.name, self.out, self.err, self.calls = name, out, err, 0

    def complete_json(self, system, user, schema, timeout=0):
        self.calls += 1
        if self.err:
            raise self.err
        return self.out

    def complete_text(self, system, user, timeout=0):
        return json.dumps(self.complete_json(system, user, {}))


def test_chain_falls_through_and_cools_down_rate_limited():
    a = _Fake("a", err=LLMError("429", "rate_limited"))
    b = _Fake("b", out={"move": "wait"})
    chain = ChainBackend([a, b], cooldown_s=600)
    assert chain.complete_json("s", "u", {}) == {"move": "wait"}
    assert chain.complete_json("s", "u", {}) == {"move": "wait"}
    assert a.calls == 1 and b.calls == 2          # a skipped while cooling down
    assert chain.last_used == "b"


def test_chain_all_failed_raises():
    chain = ChainBackend([_Fake("a", err=LLMError("x", "auth"))])
    with pytest.raises(LLMError):
        chain.complete_json("s", "u", {})


def test_claude_cli_backend_is_toolless_and_schema_enforced():
    with patch("autobot.integrations.claude_code_bridge.run_headless") as rh:
        rh.return_value = {"ok": True, "data": {"result": "{}", "structured_output": {"move": "ask_user"}}, "error": ""}
        out = ClaudeCLIBackend(model="haiku").complete_json("sys", "user", {"type": "object"})
    assert out == {"move": "ask_user"}
    kw = rh.call_args.kwargs
    assert kw["tools"] == [] and kw["json_schema"] == {"type": "object"} and kw["permission_mode"] == "plan"
    assert kw["model"] == "haiku" and kw["no_session_persistence"] is True


def test_claude_cli_backend_rate_limit_class():
    with patch("autobot.integrations.claude_code_bridge.run_headless") as rh:
        rh.return_value = {"ok": False, "data": None, "error": "usage limit", "error_class": "rate_limited"}
        with pytest.raises(LLMError) as ei:
            ClaudeCLIBackend().complete_json("s", "u", {})
    assert ei.value.error_class == "rate_limited"


def test_openai_compat_downgrades_json_mode_when_rejected():
    b = OpenAICompatBackend("x", "http://h", "k", "m")
    calls = []

    class BadRequest(Exception):
        status_code = 400

    def create(**kw):
        calls.append(kw.get("response_format"))
        if kw.get("response_format", {}).get("type") == "json_schema":
            raise BadRequest("response_format json_schema not supported")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"move": "wait"}'))])
    b._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert b.complete_json("s", "u", {"type": "object"}) == {"move": "wait"}
    assert [c["type"] for c in calls] == ["json_schema", "json_object"]
    b.complete_json("s", "u", {"type": "object"})
    assert calls[-1]["type"] == "json_object"      # remembered, doesn't retry json_schema every call


def test_shim_lets_coreloop_use_any_backend():
    shim = ChatCompletionsShim(_Fake("a", out={"x": 1}))
    r = shim.chat.completions.create(model="m", messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert json.loads(r.choices[0].message.content) == {"x": 1}


def test_factory(monkeypatch):
    from autobot.llm.factory import build_backend, get_manager_llm
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CEREBRAS_API_KEY",
              "AUTOBOT_LLM_BASE_URL", "AUTOBOT_LLM_MODEL", "AUTOBOT_OLLAMA_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AUTOBOT_MANAGER_LLM", "none")
    assert get_manager_llm() is None
    monkeypatch.setenv("AUTOBOT_MANAGER_LLM", "groq,gemini")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    chain = get_manager_llm()
    assert [b.name for b in chain.backends] == ["groq", "gemini"]
    assert chain.trains_on_data is True                       # gemini free tier is flagged
    assert [b.name for b in chain.without_training_backends().backends] == ["groq"]
    assert build_backend("openai_compat") is None             # needs base URL + model
