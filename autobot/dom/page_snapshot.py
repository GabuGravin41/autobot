"""
Page Snapshot — Hybrid perception via Chrome DevTools Protocol.

Connects to Chrome's remote debugging port (9222) to extract a structured
text snapshot of the current page without taking a screenshot.

This gives the LLM:
  - All interactive elements (buttons, links, inputs, selects) with text/labels
  - Visible page text (truncated)
  - Current URL + title from the real browser (not the emulator's cached value)

Used by AgentLoop to complement or replace screenshots:
  - Browser tab → DOM snapshot (fast, cheap, precise) + screenshot (visual context)
  - Desktop app → screenshot only (no DOM available)

Chrome must be running with --remote-debugging-port=9222.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_CDP_HOST = "localhost"
_CDP_PORT = 9222
_JS_EXTRACT = """
(function() {
    const MAX_TEXT = 3000;
    const MAX_ELEMENTS = 80;

    // ── 1. Interactive elements ──────────────────────────────────────────────
    const INTERACTIVE = 'a[href], button, input:not([type="hidden"]), select, textarea, [role="button"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"]';
    const elements = [];
    let idx = 1;

    document.querySelectorAll(INTERACTIVE).forEach(el => {
        if (idx > MAX_ELEMENTS) return;
        if (!el.offsetParent && el.tagName !== 'BODY') return; // skip hidden

        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type') || '';
        const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim().slice(0, 80);
        const href = el.getAttribute('href') || '';
        const name = el.getAttribute('name') || el.getAttribute('id') || '';

        if (!text && !href) return; // skip completely unlabelled elements

        let desc = `[${idx}] <${tag}`;
        if (type) desc += ` type="${type}"`;
        if (name) desc += ` name="${name}"`;
        desc += `>`;
        if (text) desc += ` ${text}`;
        if (href && href !== '#' && !href.startsWith('javascript')) desc += ` → ${href.slice(0, 60)}`;

        elements.push(desc);
        idx++;
    });

    // ── 2. Popup / dialog detection ──────────────────────────────────────────
    const DIALOG_SEL = '[role="dialog"], [role="alertdialog"], dialog[open], .modal, .modal-dialog, .popup, [aria-modal="true"]';
    const dialogs = [];
    document.querySelectorAll(DIALOG_SEL).forEach(d => {
        if (!d.offsetParent) return; // skip hidden dialogs
        const title = (d.querySelector('h1,h2,h3,h4,[role="heading"]') || {}).innerText || '';
        const btns = Array.from(d.querySelectorAll('button')).map(b => b.innerText.trim()).filter(Boolean).slice(0, 5);
        dialogs.push({ title: title.trim().slice(0, 80), buttons: btns });
    });

    // ── 3. Visible page text ─────────────────────────────────────────────────
    // Main content areas first, then fallback to full body
    let bodyText = '';
    const main = document.querySelector('main, [role="main"], article, #content, .content');
    const source = main || document.body;
    if (source) {
        // Remove nav/header/footer/script/style clutter
        const clone = source.cloneNode(true);
        clone.querySelectorAll('nav, header, footer, script, style, noscript').forEach(n => n.remove());
        bodyText = (clone.innerText || '').replace(/\\n{3,}/g, '\\n\\n').trim().slice(0, MAX_TEXT);
    }

    // ── 4. Page metadata ─────────────────────────────────────────────────────
    return JSON.stringify({
        url: window.location.href,
        title: document.title,
        elements: elements,
        dialogs: dialogs,
        text: bodyText,
        num_interactive: elements.length,
        num_links: document.querySelectorAll('a[href]').length,
        num_inputs: document.querySelectorAll('input, textarea, select').length,
    });
})()
"""


@dataclass
class PageSnapshot:
    url: str
    title: str
    elements: list[str]          # ["[1] <button> Login", "[2] <a> Home → /"]
    text: str                     # Visible page text (truncated)
    num_interactive: int
    num_links: int
    num_inputs: int
    dialogs: list[dict] = None   # [{"title": "...", "buttons": ["OK", "Cancel"]}]

    def __post_init__(self):
        if self.dialogs is None:
            self.dialogs = []

    @property
    def has_popup(self) -> bool:
        return bool(self.dialogs)

    def to_prompt_text(self) -> str:
        """Format for inclusion in the LLM prompt."""
        parts = [f"URL: {self.url}", f"Title: {self.title}"]

        # Popup alert — shown first so agent handles it before anything else
        if self.dialogs:
            dialog_lines = []
            for d in self.dialogs:
                t = d.get("title", "(no title)")
                btns = d.get("buttons", [])
                btn_str = " | ".join(btns) if btns else "no buttons detected"
                dialog_lines.append(f"  POPUP: \"{t}\"  →  buttons: [{btn_str}]")
            parts.insert(0,
                "⚠️  POPUP/DIALOG DETECTED — handle this before doing anything else:\n"
                + "\n".join(dialog_lines)
            )

        if self.elements:
            parts.append(
                f"\nInteractive elements ({self.num_interactive} total):\n"
                + "\n".join(self.elements)
            )
        else:
            parts.append("\n(No interactive elements detected)")

        if self.text:
            parts.append(f"\nPage text (excerpt):\n{self.text[:1500]}")

        return "\n".join(parts)


class CDPClient:
    """
    Minimal Chrome DevTools Protocol client over WebSocket.
    Uses only stdlib + the websockets package already in requirements.
    """

    def __init__(self, ws_url: str) -> None:
        self._ws_url = ws_url
        self._ws = None
        self._msg_id = 0

    async def connect(self) -> None:
        import websockets
        self._ws = await websockets.connect(self._ws_url, ping_interval=None)

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def call(self, method: str, params: dict | None = None) -> Any:
        self._msg_id += 1
        msg = json.dumps({"id": self._msg_id, "method": method, "params": params or {}})
        await self._ws.send(msg)
        # Read until we get the response for our message id
        for _ in range(20):
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            data = json.loads(raw)
            if data.get("id") == self._msg_id:
                return data.get("result", {})
        return {}


async def _get_active_tab_ws_url() -> str | None:
    """Find the WebSocket URL for the active/focused Chrome tab."""
    try:
        url = f"http://{_CDP_HOST}:{_CDP_PORT}/json"
        req = urllib.request.urlopen(url, timeout=1)
        tabs = json.loads(req.read())
        # Prefer the first 'page' type tab (active tab is usually first)
        for tab in tabs:
            if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
                return tab["webSocketDebuggerUrl"]
    except Exception:
        pass
    return None


def _get_current_url_sync() -> str | None:
    """
    Read the active tab URL from Chrome's CDP JSON endpoint (no WebSocket needed).

    This is a cheap synchronous HTTP call (~1ms) used to get the real current URL
    after the agent navigates by clicking — the HumanModeEmulator's cached _url
    only updates on goto() calls, so this gives us the truth.

    Returns None if Chrome isn't running or isn't reachable.
    """
    try:
        req = urllib.request.urlopen(
            f"http://{_CDP_HOST}:{_CDP_PORT}/json", timeout=1
        )
        tabs = json.loads(req.read())
        for tab in tabs:
            if tab.get("type") == "page" and tab.get("url"):
                return tab["url"]
    except Exception:
        pass
    return None


async def get_page_snapshot(timeout: float = 4.0) -> PageSnapshot | None:
    """
    Extract a structured snapshot of the current browser page via CDP.

    Returns None if Chrome DevTools is unavailable or the call times out.
    Never raises — always safe to call and ignore the result.
    Retries up to 2 times with short backoff for transient connection failures.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            ws_url = await asyncio.wait_for(_get_active_tab_ws_url(), timeout=1.0)
            if not ws_url:
                return None  # Chrome not running — no point retrying

            client = CDPClient(ws_url)
            await asyncio.wait_for(client.connect(), timeout=2.0)

            try:
                result = await asyncio.wait_for(
                    client.call("Runtime.evaluate", {
                        "expression": _JS_EXTRACT,
                        "returnByValue": True,
                    }),
                    timeout=timeout,
                )
            finally:
                await client.close()

            value = result.get("result", {}).get("value")
            if not value:
                return None

            data = json.loads(value)
            return PageSnapshot(
                url=data.get("url", ""),
                title=data.get("title", ""),
                elements=data.get("elements", []),
                text=data.get("text", ""),
                num_interactive=data.get("num_interactive", 0),
                num_links=data.get("num_links", 0),
                num_inputs=data.get("num_inputs", 0),
                dialogs=data.get("dialogs", []),
            )

        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))  # 0.5s, 1.0s backoff
            continue

    logger.debug(f"Page snapshot unavailable after 3 attempts: {last_error}")
    return None
