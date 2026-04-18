"""
Browser — CDP-backed helpers for reading and copying text from the active tab.

Most "copy text from the page" flows with mouse+Ctrl+A+Ctrl+C are fragile:
Ctrl+A may select the whole page instead of just the response, the cursor
might not be in the intended element, and the clipboard may silently end up
empty. This module reads directly from the DOM via CDP so the agent can
reliably capture long AI responses, code blocks, or any element text.

All methods are synchronous (they block for a short timeout) because the
agent executes them via the AST dispatcher in the main event loop via
asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_CDP_HOST = "localhost"
_CDP_PORT = 9222


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
