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
    // contenteditable="true" covers rich-text editors like Grok, ChatGPT, Overleaf
    // which do NOT use <textarea> — their .value is always empty.
    const INTERACTIVE = 'a[href], button, input:not([type="hidden"]), select, textarea, [contenteditable="true"], [role="button"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"]';
    const elements = [];
    let idx = 1;

    document.querySelectorAll(INTERACTIVE).forEach(el => {
        if (idx > MAX_ELEMENTS) return;
        if (!el.offsetParent && el.tagName !== 'BODY') return; // skip hidden

        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type') || '';
        const isContentEditable = el.getAttribute('contenteditable') === 'true';
        const isInput = (tag === 'input' || tag === 'textarea') && !isContentEditable;
        // For inputs/textareas: read .value. For contenteditable rich-text editors
        // (Grok, ChatGPT, Overleaf, etc.): read innerText — .value is always ''.
        // We MUST report the actual length or the agent will re-type content already there.
        const rawValue = isInput
            ? (el.value || '')
            : (isContentEditable ? (el.innerText || el.textContent || '') : '');
        const valueLen = rawValue.length;
        const rawText = (el.innerText || rawValue || el.placeholder || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
        const text = rawText.slice(0, 80);
        const textTruncated = rawText.length > 80;
        const href = el.getAttribute('href') || '';
        const name = el.getAttribute('name') || el.getAttribute('id') || '';

        if (!text && !href) return; // skip completely unlabelled elements

        // ── Element state flags ─────────────────────────────────────
        const flags = [];
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') flags.push('DISABLED');
        if (el.required || el.getAttribute('aria-required') === 'true') flags.push('required');
        if (el.readOnly) flags.push('readonly');
        if ((isInput || isContentEditable) && rawValue.trim()) flags.push(`filled:${valueLen}ch`);
        if (isContentEditable) flags.push('richtext');
        if (el.getAttribute('aria-expanded') === 'true') flags.push('expanded');
        if (el.getAttribute('aria-checked') === 'true') flags.push('checked');
        if (el.getAttribute('aria-busy') === 'true') flags.push('loading');
        const placeholder = el.placeholder || el.getAttribute('data-placeholder') || '';

        // Screen coordinates — agent uses these directly for mouse.click(x, y)
        const r = el.getBoundingClientRect();
        const cx = Math.round(r.left + r.width / 2);
        const cy = Math.round(r.top + r.height / 2 + window.scrollY);

        let desc = `[${idx}] <${tag}`;
        if (type) desc += ` type="${type}"`;
        if (name) desc += ` name="${name}"`;
        if (placeholder && !text.includes(placeholder)) desc += ` placeholder="${placeholder.slice(0, 40)}"`;
        desc += `>`;
        // Append click coordinates so agent never has to guess
        desc += ` @(${cx},${cy})`;
        if (flags.length) desc += ` [${flags.join(', ')}]`;
        if (text) {
            desc += ` ${text}`;
            if (textTruncated) desc += `…(truncated from ${rawText.length}ch)`;
        }
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


async def _get_active_tab_ws_url(url_hint: str | None = None) -> str | None:
    """Find the WebSocket URL for the active/focused Chrome tab.

    If url_hint is provided, prefer the tab whose URL matches it.
    This is critical in multi-tab scenarios where the first CDP tab
    may not be the one the agent just navigated to.
    """
    try:
        req = urllib.request.urlopen(f"http://{_CDP_HOST}:{_CDP_PORT}/json", timeout=1)
        tabs = json.loads(req.read())
        page_tabs = [t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
        if not page_tabs:
            return None

        if url_hint and url_hint not in ("about:blank", ""):
            hint_stripped = url_hint.rstrip("/")
            # Exact or prefix match first
            for tab in page_tabs:
                tab_url = tab.get("url", "").rstrip("/")
                if tab_url == hint_stripped or tab_url.startswith(hint_stripped):
                    return tab["webSocketDebuggerUrl"]
            # Hostname match fallback
            try:
                from urllib.parse import urlparse
                hint_host = urlparse(url_hint).netloc
                for tab in page_tabs:
                    if hint_host and hint_host in tab.get("url", ""):
                        return tab["webSocketDebuggerUrl"]
            except Exception:
                pass

        # Default: first page tab
        return page_tabs[0]["webSocketDebuggerUrl"]
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


async def get_page_snapshot(timeout: float = 4.0, url_hint: str | None = None) -> PageSnapshot | None:
    """
    Extract a structured snapshot of the current browser page via CDP.

    Returns None if Chrome DevTools is unavailable or the call times out.
    Never raises — always safe to call and ignore the result.
    Retries up to 2 times with short backoff for transient connection failures.

    url_hint: the URL the agent expects to be on. Used to pick the correct
    tab in multi-tab Chrome sessions instead of blindly using the first tab.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            ws_url = await asyncio.wait_for(_get_active_tab_ws_url(url_hint=url_hint), timeout=1.0)
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
