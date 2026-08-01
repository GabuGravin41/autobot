"""Behavioral test for autobot/agent/anthropic_adapter.py's actual message
translation logic — not just "does it construct" (test_anthropic_adapter.py
already covers that, and currently fails here since the real `anthropic`
package isn't installed in this environment).

Injects a FAKE `anthropic` module into sys.modules so this runs with zero
network access and zero real dependency — the exact class of check this
project has needed all along: proving the CODE works, not just that it reads
correctly. Also drives it through AgentLoop's real _make_llm_call() to prove
the sync-adapter-in-an-async-loop path (the await-then-TypeError-then-
to_thread fallback) actually functions, not just the adapter in isolation.
"""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


# ── Fake `anthropic` package — records what it was called with ────────────
_last_call_kwargs = {}

class _FakeMessages:
    def create(self, **kwargs):
        _last_call_kwargs.clear()
        _last_call_kwargs.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text='{"ok": true, "from": "fake-claude"}')])

class _FakeAnthropicClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = _FakeMessages()

fake_anthropic_module = ModuleType("anthropic")
fake_anthropic_module.Anthropic = _FakeAnthropicClient
sys.modules["anthropic"] = fake_anthropic_module

from autobot.agent.anthropic_adapter import get_anthropic_llm_client  # noqa: E402


def main():
    client = get_anthropic_llm_client(api_key="sk-ant-test-mock")
    check("adapter constructs with fake anthropic installed", client is not None)
    check("exposes chat.completions.create",
          client is not None and hasattr(client.chat.completions, "create"))

    # ---- system message separation ----
    resp = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "Say hi."},
        ],
        temperature=0.1,
    )
    check("system message pulled into system kwarg, not messages list",
          _last_call_kwargs.get("system") == "You are a helpful agent."
          and len(_last_call_kwargs.get("messages", [])) == 1,
          str(_last_call_kwargs))
    check("response shaped like an OpenAI ChatCompletion",
          resp.choices[0].message.content == '{"ok": true, "from": "fake-claude"}')

    # ---- response_format=json_object injects a JSON instruction ----
    client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "system", "content": "Be terse."},
                  {"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    check("response_format=json_object adds a JSON instruction to system",
          "JSON" in _last_call_kwargs.get("system", ""),
          _last_call_kwargs.get("system"))

    # ---- vision: OpenAI-style image_url content converts to Anthropic's format ----
    client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "what's in this screenshot?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
            ],
        }],
    )
    sent = _last_call_kwargs["messages"][0]["content"]
    check("vision content translated to Anthropic image block",
          any(p.get("type") == "image" and p.get("source", {}).get("media_type") == "image/png"
              and p.get("source", {}).get("data") == "QUJD" for p in sent),
          str(sent))
    check("text part preserved alongside the image",
          any(p.get("type") == "text" and "screenshot" in p.get("text", "") for p in sent),
          str(sent))

    # ---- non-Claude model name gets mapped to a real Claude model, not sent as-is ----
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
    )
    check("non-Claude model name is not sent to the Anthropic API verbatim",
          "claude" in _last_call_kwargs.get("model", "").lower(),
          _last_call_kwargs.get("model"))

    # ---- THE INTEGRATION TEST: drive it through AgentLoop's real call path ----
    # This is what actually matters: AgentLoop._make_llm_call() first tries
    # `await self.llm_client.chat.completions.create(...)`. The adapter's
    # create() is a plain sync method, so awaiting its return value raises
    # TypeError, which the loop catches and retries via
    # asyncio.to_thread(...). If that fallback didn't work, Anthropic would
    # be silently unusable through the real agent regardless of how correct
    # the adapter looks in isolation.
    async def integration():
        from autobot.agent.loop import AgentLoop
        agent = AgentLoop(page=None, llm_client=client, goal="say hi",
                          model="claude-3-5-sonnet-20241022", max_steps=1)
        from autobot.dom.models import BrowserState, DOMElementNode, SelectorMap
        empty_state = BrowserState(
            url="", title="", tabs=[], page_info=None,
            element_tree=DOMElementNode(index=None, tag_name="body", text="",
                                        attributes={}, children=[], depth=0),
            selector_map=SelectorMap(), screenshot_b64=None,
        )
        raw = await agent._make_llm_call([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ])
        return raw

    raw = asyncio.run(integration())
    check("AgentLoop's sync-fallback path works with a sync Anthropic adapter",
          raw == '{"ok": true, "from": "fake-claude"}', repr(raw))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


main()
