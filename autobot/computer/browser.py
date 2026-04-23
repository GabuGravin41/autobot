"""
Browser — CDP-backed helpers for reading, copying, and interacting with the active tab.

All interaction methods (click_element, fill, focus, read_value) operate via Chrome
DevTools Protocol so the agent can reliably interact with elements by their DOM index
from the page snapshot — no coordinate guessing required.

All methods are synchronous (they block for a short timeout) because the
agent executes them via the AST dispatcher in the main event loop via
asyncio.to_thread.

Three-layer interaction priority:
  1. browser.fill(index, text)        — CDP direct: most reliable for inputs
  2. browser.click_element(index)     — CDP click at exact current coordinates
  3. computer.mouse.click(x, y)       — visual fallback for non-DOM elements
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_CDP_HOST = "localhost"
_CDP_PORT = 9222

# Selector matching the same elements as page_snapshot._JS_EXTRACT
_INTERACTIVE_SEL = (
    'a[href], button, input:not([type="hidden"]), select, textarea, '
    '[contenteditable="true"], [role="button"], [role="link"], '
    '[role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"]'
)
# Pre-escaped version safe for embedding inside JS double-quoted strings in f-strings
# (backslash expressions are not allowed inside f-string {} in Python < 3.12)
_INTERACTIVE_SEL_JS = _INTERACTIVE_SEL.replace('"', '\\"')


def _run_sync(coro):
    """Run an async coroutine from sync context.

    Browser methods are dispatched via asyncio.to_thread(...) in the agent loop,
    so we are in a worker thread with no running event loop — asyncio.run works.
    If called from the main event loop thread, we fall back to a worker thread.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is None:
        return asyncio.run(coro)

    # We're on a thread that has a running loop — must not block it.
    result: dict[str, Any] = {}
    def _runner() -> None:
        result["value"] = asyncio.run(coro)
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=15)
    return result.get("value")


async def _cdp_eval(js: str, url_hint: str | None = None, timeout: float = 4.0) -> Any:
    """Run a JS expression in the active tab via CDP. Returns the raw value or None."""
    from autobot.dom.page_snapshot import _get_active_tab_ws_url, CDPClient
    ws_url = await asyncio.wait_for(_get_active_tab_ws_url(url_hint=url_hint), timeout=1.5)
    if not ws_url:
        return None
    client = CDPClient(ws_url)
    await asyncio.wait_for(client.connect(), timeout=2.0)
    try:
        res = await asyncio.wait_for(
            client.call("Runtime.evaluate", {
                "expression": js,
                "returnByValue": True,
            }),
            timeout=timeout,
        )
    finally:
        await client.close()
    return res.get("result", {}).get("value")


def _js_find_by_index(index: int) -> str:
    """Return a JS expression that finds the nth interactive element (1-based).

    Uses the SAME selector and hidden-element filter as page_snapshot._JS_EXTRACT
    so the index here always matches the index shown in the DOM snapshot.
    Scrolls the element into view and returns current bounding-rect center coords.
    """
    sel = _INTERACTIVE_SEL_JS
    return f"""
(function() {{
    const SEL = "{sel}";
    let idx = 1, found = null;
    for (const el of document.querySelectorAll(SEL)) {{
        const s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') continue;
        if (idx++ === {index}) {{ found = el; break; }}
    }}
    if (!found) return JSON.stringify({{ok: false, error: 'element {index} not found in DOM'}});
    found.scrollIntoView({{behavior: 'instant', block: 'center'}});
    const r = found.getBoundingClientRect();
    const tag = found.tagName.toLowerCase();
    const isCE = found.getAttribute('contenteditable') === 'true';
    const isInp = tag === 'input' || tag === 'textarea';
    const val = isInp ? (found.value || '') : isCE ? (found.innerText || found.textContent || '') : '';
    return JSON.stringify({{
        ok: true,
        x: Math.round(r.left + r.width / 2),
        y: Math.round(r.top + r.height / 2),
        tag: tag,
        ce: isCE,
        is_input: isInp,
        value: val.slice(0, 300),
        type: found.getAttribute('type') || '',
        name: found.getAttribute('name') || found.getAttribute('id') || ''
    }});
}})()
"""


class Browser:
    """CDP-backed text read / copy helpers for the active Chrome tab."""

    def read(self, selector: str) -> str:
        """Return the innerText of the first element matching the CSS selector.

        Use this to capture AI chat responses, code blocks, or any element's text
        without the fragile Ctrl+A + Ctrl+C dance. Example selectors for common sites:
            read('.message-content')           — Grok/ChatGPT response bubble
            read('[data-message-author-role="assistant"]')  — ChatGPT assistant messages
            read('main article')                — article body
        Returns an empty string if no element matches or CDP is unavailable.
        """
        js = (
            f"(function() {{"
            f"  var el = document.querySelector({json.dumps(selector)});"
            f"  return el ? (el.innerText || el.textContent || '') : '';"
            f"}})()"
        )
        try:
            val = _run_sync(_cdp_eval(js))
            return str(val) if val is not None else ""
        except Exception as e:
            logger.warning(f"browser.read({selector!r}) failed: {e}")
            return ""

    def read_all(self, selector: str, separator: str = "\n\n") -> str:
        """Return the innerText of ALL matching elements joined by separator.

        Useful when an AI response is split across multiple message bubbles
        (e.g. ChatGPT's streaming output creates multiple DOM nodes).
        """
        js = (
            f"(function() {{"
            f"  var els = document.querySelectorAll({json.dumps(selector)});"
            f"  return Array.from(els).map(e => e.innerText || e.textContent || '').join({json.dumps(separator)});"
            f"}})()"
        )
        try:
            val = _run_sync(_cdp_eval(js))
            return str(val) if val is not None else ""
        except Exception as e:
            logger.warning(f"browser.read_all({selector!r}) failed: {e}")
            return ""

    def copy(self, selector: str) -> str:
        """Read element text via CDP and write it to the system clipboard in one step.

        Returns the copied text (or empty string if nothing was copied). This is the
        reliable replacement for "click → ctrl+a → ctrl+c" when the target element
        is in the page DOM. For AI chat responses prefer this over selection+copy.
        """
        text = self.read_all(selector)
        if not text:
            logger.info(f"browser.copy({selector!r}): no text found")
            return ""
        try:
            from autobot.computer.clipboard import Clipboard
            Clipboard().set(text)
            logger.info(f"browser.copy({selector!r}): copied {len(text)} chars to clipboard")
        except Exception as e:
            logger.warning(f"browser.copy: clipboard.set failed: {e}")
        return text

    def url(self) -> str:
        """Return the URL of the active tab, read straight from CDP."""
        try:
            val = _run_sync(_cdp_eval("window.location.href"))
            return str(val) if val else ""
        except Exception:
            return ""

    def title(self) -> str:
        """Return the document title of the active tab."""
        try:
            val = _run_sync(_cdp_eval("document.title"))
            return str(val) if val else ""
        except Exception:
            return ""

    # ── Element interaction by DOM index ────────────────────────────────────────
    # The index matches the [N] number shown in the page snapshot DOM listing.
    # These are the PRIMARY way to interact with browser elements — always prefer
    # these over coordinate-guessing with computer.mouse.click(x, y).

    def click_element(self, index: int) -> str:
        """Click a DOM element by its snapshot index (the [N] number in the DOM listing).

        Scrolls the element into view, gets its CURRENT bounding-rect center from
        CDP (not the stale snapshot coords), then fires real mouse events via CDP.
        This is the most reliable way to click links, buttons, and inputs.

        Returns a status string: 'clicked [N] <tag>' or an error message.
        Use this instead of computer.mouse.click() whenever a DOM index is available.

        Example: browser.click_element(3)  ← clicks element [3] from DOM snapshot
        """
        return _run_sync(self._click_element_async(index)) or f"click_element({index}) failed"

    async def _click_element_async(self, index: int) -> str:
        from autobot.dom.page_snapshot import _get_active_tab_ws_url, CDPClient
        try:
            ws_url = await asyncio.wait_for(_get_active_tab_ws_url(), timeout=1.5)
            if not ws_url:
                return "CDP unavailable — Chrome not running with --remote-debugging-port=9222"
            client = CDPClient(ws_url)
            await asyncio.wait_for(client.connect(), timeout=2.0)
            try:
                # Find element and get current coordinates
                res = await asyncio.wait_for(
                    client.call("Runtime.evaluate", {
                        "expression": _js_find_by_index(index),
                        "returnByValue": True,
                    }),
                    timeout=3.0,
                )
                raw = res.get("result", {}).get("value")
                if not raw:
                    return f"element [{index}] not found (no CDP response)"
                info = json.loads(raw)
                if not info.get("ok"):
                    return f"element [{index}] not found: {info.get('error', 'unknown')}"

                cx, cy = info["x"], info["y"]
                tag = info.get("tag", "?")

                # Dispatch real mouse events at exact CDP coordinates
                await client.call("Input.dispatchMouseEvent", {
                    "type": "mouseMoved", "x": cx, "y": cy, "button": "none", "clickCount": 0
                })
                await asyncio.sleep(0.05)
                await client.call("Input.dispatchMouseEvent", {
                    "type": "mousePressed", "x": cx, "y": cy, "button": "left", "clickCount": 1
                })
                await client.call("Input.dispatchMouseEvent", {
                    "type": "mouseReleased", "x": cx, "y": cy, "button": "left", "clickCount": 1
                })

                logger.info(f"browser.click_element({index}): clicked <{tag}> at ({cx},{cy})")
                return f"clicked [{index}] <{tag}> at ({cx},{cy})"

            finally:
                await client.close()

        except Exception as e:
            logger.warning(f"browser.click_element({index}) failed: {e}")
            return f"error: {e}"

    def fill(self, index: int, text: str) -> str:
        """Type text into a DOM input/textarea/contenteditable by its snapshot index.

        This is the MOST RELIABLE way to type into any form field. It:
          1. Finds the element by index (same [N] from DOM snapshot)
          2. Scrolls it into view and clicks to focus
          3. Clears any existing content
          4. Inserts text via CDP Input.insertText (fires real browser events)
          5. Verifies the text appeared and returns confirmation

        Always prefer this over: computer.keyboard.type() after clicking.
        For contenteditable rich-text editors (ChatGPT, Grok, etc.) this also works.

        Example: browser.fill(2, "hello world")  ← types into element [2]
        Returns: 'filled [N]: typed X chars, verified: True/False (actual: "...")'
        """
        result = _run_sync(self._fill_async(index, text))
        return result or f"fill({index}) failed with no response"

    async def _fill_async(self, index: int, text: str) -> str:
        from autobot.dom.page_snapshot import _get_active_tab_ws_url, CDPClient
        try:
            ws_url = await asyncio.wait_for(_get_active_tab_ws_url(), timeout=1.5)
            if not ws_url:
                return "CDP unavailable"
            client = CDPClient(ws_url)
            await asyncio.wait_for(client.connect(), timeout=2.0)
            try:
                # Step 1: find element and scroll into view
                find_res = await asyncio.wait_for(
                    client.call("Runtime.evaluate", {
                        "expression": _js_find_by_index(index),
                        "returnByValue": True,
                    }),
                    timeout=3.0,
                )
                raw = find_res.get("result", {}).get("value")
                if not raw:
                    return f"element [{index}] not found"
                info = json.loads(raw)
                if not info.get("ok"):
                    return f"element [{index}] not found: {info.get('error')}"

                cx, cy = info["x"], info["y"]
                is_input = info.get("is_input", False)
                is_ce = info.get("ce", False)
                tag = info.get("tag", "?")

                # Step 2: click to focus
                await client.call("Input.dispatchMouseEvent", {
                    "type": "mouseMoved", "x": cx, "y": cy, "button": "none", "clickCount": 0
                })
                await client.call("Input.dispatchMouseEvent", {
                    "type": "mousePressed", "x": cx, "y": cy, "button": "left", "clickCount": 1
                })
                await client.call("Input.dispatchMouseEvent", {
                    "type": "mouseReleased", "x": cx, "y": cy, "button": "left", "clickCount": 1
                })
                await asyncio.sleep(0.15)

                # Step 3: clear existing content
                if is_input:
                    clear_js = f"""
(function() {{
    const SEL = "{_INTERACTIVE_SEL_JS}";
    let idx = 1, found = null;
    for (const el of document.querySelectorAll(SEL)) {{
        const s = window.getComputedStyle(el);
        if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0') continue;
        if (idx++==={index}) {{ found=el; break; }}
    }}
    if (!found) return 'not found';
    found.focus();
    if (found.select) found.select();
    found.value = '';
    found.dispatchEvent(new Event('input', {{bubbles:true}}));
    found.dispatchEvent(new Event('change', {{bubbles:true}}));
    return 'cleared';
}})()"""
                else:
                    # contenteditable (ChatGPT, Grok, rich-text editors)
                    clear_js = f"""
(function() {{
    const SEL = "{_INTERACTIVE_SEL_JS}";
    let idx = 1, found = null;
    for (const el of document.querySelectorAll(SEL)) {{
        const s = window.getComputedStyle(el);
        if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0') continue;
        if (idx++==={index}) {{ found=el; break; }}
    }}
    if (!found) return 'not found';
    found.focus();
    document.execCommand('selectAll');
    document.execCommand('delete');
    return 'cleared';
}})()"""

                await asyncio.wait_for(
                    client.call("Runtime.evaluate", {"expression": clear_js, "returnByValue": True}),
                    timeout=2.0,
                )
                await asyncio.sleep(0.1)

                # Step 4: insert text via CDP (fires real input events, works in all editors)
                await asyncio.wait_for(
                    client.call("Input.insertText", {"text": text}),
                    timeout=max(5.0, len(text) / 200),
                )
                await asyncio.sleep(0.2)

                # Step 5: verify text appeared
                verify_res = await asyncio.wait_for(
                    client.call("Runtime.evaluate", {
                        "expression": _js_find_by_index(index),
                        "returnByValue": True,
                    }),
                    timeout=2.0,
                )
                raw2 = verify_res.get("result", {}).get("value")
                after_val = ""
                if raw2:
                    after_info = json.loads(raw2)
                    after_val = after_info.get("value", "")

                # Consider verified if our text (or significant portion) is in the field
                verified = text[:30] in after_val or len(after_val) >= min(len(text), 5)
                preview = after_val[:80].replace("\n", "↵")

                logger.info(f"browser.fill({index}): typed {len(text)} chars into <{tag}>, verified={verified}")
                return f"filled [{index}] <{tag}>: typed {len(text)} chars, verified: {verified} (actual: \"{preview}\")"

            finally:
                await client.close()

        except Exception as e:
            logger.warning(f"browser.fill({index}) failed: {e}")
            return f"error: {e}"

    def read_value(self, index: int) -> str:
        """Read the current value or text of a DOM element by its snapshot index.

        For inputs/textareas: returns .value (what's been typed so far).
        For contenteditable elements: returns .innerText.
        Use this to verify that text was actually typed, or to check current field state.

        Example: browser.read_value(2)  ← reads current content of element [2]
        """
        try:
            raw = _run_sync(_cdp_eval(_js_find_by_index(index)))
            if not raw:
                return ""
            info = json.loads(raw)
            return info.get("value", "") if info.get("ok") else f"error: {info.get('error')}"
        except Exception as e:
            logger.warning(f"browser.read_value({index}) failed: {e}")
            return ""

    def focus(self, index: int) -> str:
        """Focus a DOM element by its snapshot index without simulating a mouse click.

        Useful for triggering dropdown menus or making an element active before
        pressing keyboard shortcuts (Enter, Tab, arrow keys, etc.).

        Example: browser.focus(5)  ← focuses element [5]
        """
        js = f"""
(function() {{
    const SEL = "{_INTERACTIVE_SEL_JS}";
    let idx = 1, found = null;
    for (const el of document.querySelectorAll(SEL)) {{
        const s = window.getComputedStyle(el);
        if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0') continue;
        if (idx++==={index}) {{ found=el; break; }}
    }}
    if (!found) return 'not found';
    found.scrollIntoView({{behavior:'instant',block:'center'}});
    found.focus();
    return 'focused ' + found.tagName.toLowerCase();
}})()"""
        try:
            val = _run_sync(_cdp_eval(js))
            return str(val) if val else f"element [{index}] not found"
        except Exception as e:
            logger.warning(f"browser.focus({index}) failed: {e}")
            return f"error: {e}"

    def wait_for(self, selector: str, timeout: float = 10.0) -> bool:
        """Wait until a CSS selector matches at least one visible element (up to timeout seconds).

        Useful after navigation or form submission to confirm the page has settled.
        Returns True if element appeared within timeout, False if it timed out.

        Example: browser.wait_for('button.submit', 5.0)
        Example: browser.wait_for('[data-status="done"]', 30.0)
        """
        js = f"""
(function() {{
    const el = document.querySelector({json.dumps(selector)});
    if (!el) return false;
    const s = window.getComputedStyle(el);
    return s.display !== 'none' && s.visibility !== 'hidden';
}})()"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                val = _run_sync(_cdp_eval(js, timeout=2.0))
                if val:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        logger.info(f"browser.wait_for({selector!r}): timed out after {timeout}s")
        return False

    def is_generating(self) -> bool:
        """Return True if an AI chatbot is currently generating a response.

        Checks for stop/regenerate buttons that appear during generation on common
        AI chat sites (ChatGPT, Claude, Grok, Gemini, DeepSeek, Perplexity).
        Use this to wait for generation to finish before reading the response.

        Example:
            browser.wait_for('[data-message-author-role="assistant"]', 60)
            while browser.is_generating():
                computer.anti_sleep()
        """
        js = """
(function() {
    // Stop buttons present during generation on major AI chat sites
    const stopSelectors = [
        'button[aria-label*="Stop"]',
        'button[aria-label*="stop"]',
        'button[data-testid*="stop"]',
        '[aria-label="Stop generating"]',
        'button.stop-button',
        '[class*="stop-generating"]',
        '[class*="stopButton"]',
        'button svg[class*="stop"]',
        // Streaming indicator classes
        '[data-is-streaming="true"]',
        '[class*="streaming"]',
        '.result-streaming',
    ];
    for (const sel of stopSelectors) {
        const el = document.querySelector(sel);
        if (el) {
            const s = window.getComputedStyle(el);
            if (s.display !== 'none' && s.visibility !== 'hidden') return true;
        }
    }
    return false;
})()"""
        try:
            val = _run_sync(_cdp_eval(js, timeout=2.0))
            return bool(val)
        except Exception:
            return False

    def scroll_to(self, index: int) -> str:
        """Scroll a DOM element into view by its snapshot index.

        Use this before reading an element or when you need to make an element
        visible before taking a screenshot for visual verification.

        Example: browser.scroll_to(10)  ← scrolls element [10] into view
        """
        js = f"""
(function() {{
    const SEL = "{_INTERACTIVE_SEL_JS}";
    let idx = 1, found = null;
    for (const el of document.querySelectorAll(SEL)) {{
        const s = window.getComputedStyle(el);
        if (s.display==='none'||s.visibility==='hidden'||s.opacity==='0') continue;
        if (idx++==={index}) {{ found=el; break; }}
    }}
    if (!found) return 'element {index} not found';
    found.scrollIntoView({{behavior:'smooth',block:'center'}});
    return 'scrolled to ' + found.tagName.toLowerCase() + ' [{index}]';
}})()"""
        try:
            val = _run_sync(_cdp_eval(js))
            return str(val) if val else f"element [{index}] not found"
        except Exception as e:
            return f"error: {e}"
