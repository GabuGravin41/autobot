"""Behavioral test for the multi-tab CDP targeting fix.

Root cause this guards against: computer/browser.py's click_element(),
fill(), click_via_js(), and scroll_to() called dom/page_snapshot.py's
_get_active_tab_ws_url() with NO url_hint — despite that function's own
docstring explicitly warning "the first CDP tab may not be the one the
agent just navigated to." With no hint, Chrome's /json endpoint's tab
ORDER decided which tab got acted on, typically the first one opened.

Invisible with one tab. The moment a second tab exists — exactly the
Grok-then-Overleaf shape of the benchmark this project keeps testing —
every click/fill/scroll_to/click_via_js call would silently keep
operating on the first (Grok) tab regardless of which tab the agent had
actually navigated to. That looks exactly like "the agent is clicking
blind" from the outside, without ever raising an error to explain why.

Two things are verified here, since this bug had two layers:
1. _get_active_tab_ws_url() actually picks the right tab when given a
   hint, against multiple candidate tabs (mocked /json response — no
   real Chrome needed).
2. Every computer/browser.py method that resolves an element by index
   now actually PASSES a url_hint through, rather than calling the
   hint-aware function with no hint.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


# ── Part 1: _get_active_tab_ws_url's own matching logic ────────────────────
import json as _json
import urllib.request as _urllib_request

_FAKE_TABS = [
    {"type": "page", "url": "https://grok.com/chat/abc", "webSocketDebuggerUrl": "ws://grok-tab"},
    {"type": "page", "url": "https://www.overleaf.com/project/xyz", "webSocketDebuggerUrl": "ws://overleaf-tab"},
]

def _fake_urlopen(url, timeout=1):
    resp = MagicMock()
    resp.read.return_value = _json.dumps(_FAKE_TABS).encode()
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda *a: None
    return resp


def test_tab_matching():
    from autobot.dom import page_snapshot as ps

    with patch.object(ps.urllib.request, "urlopen", side_effect=_fake_urlopen):
        # No hint -> first tab in the list (the old, buggy default behavior)
        result_no_hint = asyncio.run(ps._get_active_tab_ws_url())
        check("no hint returns the FIRST tab (documents the old failure mode)",
              result_no_hint == "ws://grok-tab", result_no_hint)

        # With a hint matching the SECOND tab -> must return the second tab,
        # not the first. This is the exact scenario: agent has navigated to
        # Overleaf (second-opened tab) and must act there, not on Grok.
        result_overleaf = asyncio.run(
            ps._get_active_tab_ws_url(url_hint="https://www.overleaf.com/project/xyz")
        )
        check("hint correctly selects the SECOND (Overleaf) tab, not the first",
              result_overleaf == "ws://overleaf-tab", result_overleaf)

        result_grok = asyncio.run(
            ps._get_active_tab_ws_url(url_hint="https://grok.com/chat/abc")
        )
        check("hint correctly selects the Grok tab when that's current",
              result_grok == "ws://grok-tab", result_grok)


# ── Part 2: every element-resolving Browser method now passes a hint ───────
def test_browser_methods_pass_hint():
    from autobot.computer.browser import Browser

    captured_hints = []

    async def fake_get_ws_url(url_hint=None):
        captured_hints.append(url_hint)
        return None  # short-circuit before any real network/CDP call

    b = Browser()
    with patch("autobot.dom.page_snapshot._get_active_tab_ws_url", side_effect=fake_get_ws_url):
        b.click_element(3, url_hint="https://www.overleaf.com/project/xyz")
        b.fill(2, "hello", url_hint="https://www.overleaf.com/project/xyz")
        b.click_via_js(5, url_hint="https://www.overleaf.com/project/xyz")
        b.scroll_to(7, url_hint="https://www.overleaf.com/project/xyz")

    check("click_element passes its url_hint through", captured_hints[0] == "https://www.overleaf.com/project/xyz")
    check("fill passes its url_hint through", captured_hints[1] == "https://www.overleaf.com/project/xyz")
    check("click_via_js passes its url_hint through", captured_hints[2] == "https://www.overleaf.com/project/xyz")
    check("scroll_to passes its url_hint through", captured_hints[3] == "https://www.overleaf.com/project/xyz")

    # Regression guard: calling with NO hint must still work (backward compat)
    # but should be visibly None, not silently defaulted to something wrong.
    captured_hints.clear()
    with patch("autobot.dom.page_snapshot._get_active_tab_ws_url", side_effect=fake_get_ws_url):
        b.click_element(3)
    check("omitting url_hint still works and is None (not silently guessed)",
          captured_hints == [None], captured_hints)


# ── Part 3: AgentLoop actually threads browser_state.url through ───────────
def test_agent_loop_passes_state_url():
    import inspect
    from autobot.agent import loop as loop_module

    src_click = inspect.getsource(loop_module.AgentLoop._execute_click)
    src_ladder = inspect.getsource(loop_module.AgentLoop._click_fallback_ladder)
    src_input = inspect.getsource(loop_module.AgentLoop._execute_input)

    check("_execute_click passes browser_state.url to click_element",
          "click_element, index, browser_state.url" in src_click.replace("\n", " ").replace("  ", " ")
          or "browser_state.url" in src_click)
    check("_click_fallback_ladder passes browser_state.url to scroll_to/click_element/click_via_js",
          src_ladder.count("browser_state.url") >= 3, f"found {src_ladder.count('browser_state.url')} occurrences")
    check("_execute_input passes browser_state.url to fill",
          "browser_state.url" in src_input)


test_tab_matching()
test_browser_methods_pass_hint()
test_agent_loop_passes_state_url()

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
