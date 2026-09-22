/**
 * Autobot Extension — Content Script
 * Injected into every page. Two independent jobs:
 *
 *  1. DOM command bridge — executes read_text / list_elements / click_index /
 *     type_index / paste_text commands relayed from the Python backend via
 *     background.js. This is what lets Autobot interact with the actual
 *     page DOM without CDP. Registered unconditionally, every time this
 *     script runs — see the comment on `__autobotBridgeInstalled` below for
 *     why that matters.
 *
 *  2. The floating overlay — a real chat with Autobot, not a one-shot
 *     "type a goal, click Run" form. Shows page context, lets you talk to
 *     Autobot conversationally (POST /api/chat) and start the plan it
 *     proposes, keep steering it once it's running (POST /api/agent/message),
 *     and respond to approval requests inline. Only built once per page
 *     (guarded by the #autobot-overlay element already existing).
 *
 * On type_index vs. paste_text: type_index sets an <input>/<textarea>'s
 * value directly (via the native property setter) and fires input/change —
 * correct and sufficient for plain form fields, but NOT for rich editors
 * built on a virtual document model (CodeMirror, Monaco, ProseMirror —
 * Overleaf's LaTeX editor is CodeMirror). Those editors own their content
 * as internal state and resync the DOM from it; writing textContent
 * directly either gets silently overwritten on the editor's next render or
 * leaves it desynced. paste_text instead dispatches a synthetic `paste`
 * ClipboardEvent, which is the event these editors actually listen for and
 * handle as a real edit through their normal insertion path — the same
 * mechanism a human's Ctrl+V goes through. This is the standard technique
 * for scripted input into this class of editor, but it is UNPROVEN here
 * against the real Overleaf editor — it hasn't been run against a live
 * Overleaf document yet. Treat it as the documented first thing to try,
 * not a guarantee; if a paste event doesn't register, the fallback is a
 * real OS-level paste (computer.clipboard.set(text) then a native Ctrl+V
 * via computer.keyboard.press, after browser_click focuses the editor),
 * which behaves exactly like a human pasting because it IS the same OS
 * mechanism.
 */

// ── DOM command bridge ───────────────────────────────────────────────────────
//
// This block is deliberately OUTSIDE any "already injected?" guard and runs
// every time this file executes. Reason: when the extension is reloaded
// (chrome://extensions → reload, which is exactly what happens after an
// update like this one), any content script already running in an
// already-open tab becomes orphaned — its chrome.runtime connection is
// invalidated, so background.js's sendMessage to it silently fails.
// background.js's fallback for that is to re-inject content.js via
// chrome.scripting.executeScript. If listener registration were gated on
// "does #autobot-overlay already exist in the DOM", that re-injection would
// find the OLD (orphaned) overlay still sitting in the page, skip
// registration, and the bridge would silently never work in that tab. The
// `__autobotBridgeInstalled` flag guards only against double-registering
// within the SAME live script instance (harmless but wasteful), never
// against a fresh instance replacing an orphaned one.
if (!window.__autobotBridgeInstalled) {
    window.__autobotBridgeInstalled = true;

    const AUTOBOT_INTERACTIVE_SELECTOR =
        'a[href], button, input, textarea, select, summary, ' +
        '[role="button"], [role="link"], [role="tab"], [role="menuitem"], ' +
        '[onclick], [tabindex]:not([tabindex="-1"]), ' +
        // CodeMirror/Monaco/ProseMirror-style rich editors (Overleaf's LaTeX
        // editor among them) render their editable surface as a
        // contenteditable div, not an <input>/<textarea> — without this
        // they'd never show up in list_elements at all, so paste_text
        // and browser_click would have nothing to target them with.
        '[contenteditable="true"], [contenteditable=""]';

    const autobotIsVisible = (el) => {
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === "hidden" || style.display === "none") return false;
        if (parseFloat(style.opacity) === 0) return false;
        return true;
    };

    const autobotElementLabel = (el) => {
        const raw = el.innerText || el.value || el.placeholder ||
            el.getAttribute("aria-label") || el.getAttribute("title") || "";
        return raw.trim().replace(/\s+/g, " ").slice(0, 80);
    };

    // Same selector + order every call, so an index from list_elements
    // still points at the same element when click_index/type_index runs
    // right after — as long as the page hasn't changed in between.
    const autobotCollectInteractiveElements = () => {
        const nodes = Array.from(document.querySelectorAll(AUTOBOT_INTERACTIVE_SELECTOR));
        const items = [];
        for (const el of nodes) {
            if (el.closest && el.closest("#autobot-overlay")) continue; // never target our own UI
            if (!autobotIsVisible(el)) continue;
            items.push(el);
            if (items.length >= 80) break;
        }
        return items;
    };

    const autobotNativeValueSetter = (el) => {
        const proto = el instanceof HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        return Object.getOwnPropertyDescriptor(proto, "value")?.set;
    };

    const autobotRunDomCommand = (command) => {
        const type = command && command.type;
        const params = (command && command.params) || {};

        switch (type) {
            case "read_text":
                return {
                    url: window.location.href,
                    title: document.title,
                    text: (document.body || document.documentElement).innerText.slice(0, 8000),
                };

            case "list_elements": {
                const els = autobotCollectInteractiveElements();
                return {
                    url: window.location.href,
                    title: document.title,
                    elements: els.map((el, i) => ({
                        index: i + 1,
                        tag: el.tagName.toLowerCase(),
                        text: autobotElementLabel(el),
                    })),
                };
            }

            case "click_index": {
                const els = autobotCollectInteractiveElements();
                const el = els[Number(params.index) - 1];
                if (!el) throw new Error(`No visible element at index ${params.index} (found ${els.length}). Call browser_list again first.`);
                el.scrollIntoView({ block: "center", behavior: "instant" });
                el.click();
                return { clicked: autobotElementLabel(el), tag: el.tagName.toLowerCase() };
            }

            case "type_index": {
                const els = autobotCollectInteractiveElements();
                const el = els[Number(params.index) - 1];
                if (!el) throw new Error(`No visible element at index ${params.index} (found ${els.length}). Call browser_list again first.`);
                el.scrollIntoView({ block: "center", behavior: "instant" });
                el.focus();
                const text = params.text || "";
                if ("value" in el) {
                    const setter = autobotNativeValueSetter(el);
                    if (setter) setter.call(el, text); else el.value = text;
                    el.dispatchEvent(new Event("input", { bubbles: true }));
                    el.dispatchEvent(new Event("change", { bubbles: true }));
                } else {
                    el.textContent = text; // contenteditable-style elements
                    el.dispatchEvent(new Event("input", { bubbles: true }));
                }
                return { typed_into: autobotElementLabel(el), tag: el.tagName.toLowerCase() };
            }

            case "paste_text": {
                // See the module docstring at the top of this file for why
                // this exists separately from type_index: CodeMirror/Monaco/
                // ProseMirror-style editors (Overleaf's LaTeX editor) manage
                // their own document state and don't reliably pick up a
                // direct textContent/value write. They DO listen for paste
                // events and handle them through their real insertion path,
                // so that's what this dispatches — a synthetic ClipboardEvent
                // carrying the text, exactly like a human's Ctrl+V.
                const els = autobotCollectInteractiveElements();
                const el = els[Number(params.index) - 1];
                if (!el) throw new Error(`No visible element at index ${params.index} (found ${els.length}). Call browser_list again first.`);
                el.scrollIntoView({ block: "center", behavior: "instant" });
                el.focus();
                const text = params.text || "";

                let dataTransfer;
                try {
                    dataTransfer = new DataTransfer();
                } catch (_) {
                    // Some contexts restrict constructing DataTransfer directly.
                    dataTransfer = new ClipboardEvent("").clipboardData;
                }
                dataTransfer.setData("text/plain", text);

                const pasteEvent = new ClipboardEvent("paste", {
                    bubbles: true,
                    cancelable: true,
                    clipboardData: dataTransfer,
                });
                el.dispatchEvent(pasteEvent);

                return {
                    pasted_into: autobotElementLabel(el),
                    tag: el.tagName.toLowerCase(),
                    chars: text.length,
                    note: "Dispatched a synthetic paste event — verify the content actually landed (browser_text or browser_list) before trusting it. If nothing changed, this editor may need a real OS-level paste instead (computer.clipboard.set + Ctrl+V via computer.keyboard.press, after clicking into the editor).",
                };
            }

            default:
                throw new Error(`Unknown DOM command type: '${type}'`);
        }
    };

    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type !== "AUTOBOT_RUN_DOM_COMMAND") return; // not ours — let other listeners handle it
        try {
            sendResponse({ ok: true, data: autobotRunDomCommand(message.command) });
        } catch (e) {
            sendResponse({ ok: false, error: String((e && e.message) || e) });
        }
        return true;
    });
}

// ── Floating overlay: a real chat with Autobot ───────────────────────────────
//  - Shows the current page context (URL + title)
//  - You type a message any time; Autobot replies (POST /api/chat) and,
//    once it understands the task, proposes a plan you can start with one
//    click — instead of having to phrase a single deterministic goal up
//    front.
//  - Once a plan is running, you can keep typing — messages are pushed into
//    the *running* agent as a mid-flight instruction (POST /api/agent/message)
//    instead of silently doing nothing, so you can redirect it the way
//    you'd redirect a person.
//  - Approval requests (e.g. before an irreversible action) show up as a
//    card right in the conversation with Allow/Block buttons, instead of
//    requiring you to find the separate popup dashboard.
//  - The step-by-step execution trace still streams in via WebSocket, but
//    as compact inline lines rather than a separate opaque log pane — so
//    you see the conversation and what Autobot is actually doing in one
//    continuous feed.

(function () {
    "use strict";

    // Avoid building the overlay twice (e.g. if this file somehow runs
    // again in a script context that already has one).
    if (document.getElementById("autobot-overlay")) return;

    // ── Build the overlay DOM ────────────────────────────────────────────────
    const overlay = document.createElement("div");
    overlay.id = "autobot-overlay";
    overlay.innerHTML = `
    <div id="autobot-panel">
      <div id="autobot-header">
        <div id="autobot-header-left">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="white">
            <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7H3a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13a.5.5 0 0 0-.5.5.5.5 0 0 0 .5.5.5.5 0 0 0 .5-.5.5.5 0 0 0-.5-.5m9 0a.5.5 0 0 0-.5.5.5.5 0 0 0 .5.5.5.5 0 0 0 .5-.5.5.5 0 0 0-.5-.5M3 21v-1a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4v1H3Z"/>
          </svg>
          <div>
            <div id="autobot-header-title">Autobot</div>
            <div id="autobot-header-status">idle</div>
          </div>
        </div>
        <div id="autobot-header-actions">
           <button id="autobot-mini-peek-toggle" title="Show a live screenshot of what Autobot is doing while a task runs">🖼️</button>
           <button id="autobot-close-btn" title="Minimize">✕</button>
        </div>
      </div>
      <div id="autobot-mini-peek" class="hidden">
        <img id="autobot-peek-img" src="" alt="Agent View" />
        <div id="autobot-peek-overlay">LIVE</div>
        <div id="autobot-peek-empty">Nothing to show yet — this fills in with a live screenshot once a task is running.</div>
      </div>
      <div id="autobot-page-context">📄 Loading page context...</div>
      <div id="autobot-chat"><div class="autobot-chat-empty">Tell Autobot what you'd like to do on this page — it's a conversation, not a form. Ask questions, describe the task loosely, and it'll ask what it needs to know before proposing a plan you can start.</div></div>
      <div id="autobot-run-banner" class="hidden">
        <span id="autobot-run-banner-text">Running…</span>
        <button id="autobot-stop-btn" title="Cancel the running task">■ Stop</button>
      </div>
      <div id="autobot-input-area">
        <textarea id="autobot-goal-input" placeholder="Message Autobot… (Enter to send, Shift+Enter for a new line)" rows="1"></textarea>
        <button id="autobot-run-btn" class="autobot-btn" title="Send">➤</button>
      </div>
    </div>
    <button id="autobot-fab" title="Open Autobot">
      <svg viewBox="0 0 24 24">
        <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7H3a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13a.5.5 0 0 0-.5.5.5.5 0 0 0 .5.5.5.5 0 0 0 .5-.5.5.5 0 0 0-.5-.5m9 0a.5.5 0 0 0-.5.5.5.5 0 0 0 .5.5.5.5 0 0 0 .5-.5.5.5 0 0 0-.5-.5M3 21v-1a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4v1H3Z"/>
      </svg>
    </button>
  `;
    document.body.appendChild(overlay);

    // ── Element refs ─────────────────────────────────────────────────────────
    const panel = document.getElementById("autobot-panel");
    const fab = document.getElementById("autobot-fab");
    const closeBtn = document.getElementById("autobot-close-btn");
    const chatEl = document.getElementById("autobot-chat");
    const statusEl = document.getElementById("autobot-header-status");
    const ctxEl = document.getElementById("autobot-page-context");
    const goalInput = document.getElementById("autobot-goal-input");
    const runBtn = document.getElementById("autobot-run-btn");
    const stopBtn = document.getElementById("autobot-stop-btn");
    const runBanner = document.getElementById("autobot-run-banner");
    const runBannerText = document.getElementById("autobot-run-banner-text");
    const peekToggle = document.getElementById("autobot-mini-peek-toggle");
    const miniPeek = document.getElementById("autobot-mini-peek");
    const peekImg = document.getElementById("autobot-peek-img");
    const header = document.getElementById("autobot-header");

    // ── Draggable overlay ────────────────────────────────────────────────────
    // #autobot-overlay ships fixed at bottom:24/right:24 (styles.css) so it
    // has a sane default position, but that corner is exactly where a lot of
    // pages put their own floating action buttons or "next" controls — the
    // overlay was sitting on top of them with no way to move out of the way.
    // Dragging by the header (panel open) or the FAB itself (panel closed)
    // switches positioning to top/left, computed from the overlay's current
    // on-screen rect at drag-start so there's no jump, and the chosen spot
    // is remembered via chrome.storage.sync so it survives reloads/restarts.
    function clampToViewport(left, top) {
        const rect = overlay.getBoundingClientRect();
        const maxLeft = Math.max(0, window.innerWidth - rect.width);
        const maxTop = Math.max(0, window.innerHeight - rect.height);
        return {
            left: Math.min(Math.max(0, left), maxLeft),
            top: Math.min(Math.max(0, top), maxTop),
        };
    }

    function applyPosition(left, top) {
        const clamped = clampToViewport(left, top);
        overlay.style.left = `${clamped.left}px`;
        overlay.style.top = `${clamped.top}px`;
        overlay.style.right = "auto";
        overlay.style.bottom = "auto";
    }

    let dragState = null; // {startX, startY, startLeft, startTop} while pointer is down
    let didDrag = false;  // crosses true once movement passes the click-vs-drag threshold

    function startDrag(e) {
        // Don't hijack clicks on the header's own buttons (peek toggle, close) —
        // only an empty-header or FAB-body press should start a drag.
        if (e.target.closest && e.target.closest("button")) return;
        const rect = overlay.getBoundingClientRect();
        dragState = { startX: e.clientX, startY: e.clientY, startLeft: rect.left, startTop: rect.top };
        didDrag = false;
        document.addEventListener("pointermove", onDrag);
        document.addEventListener("pointerup", stopDrag, { once: true });
    }

    function onDrag(e) {
        if (!dragState) return;
        const dx = e.clientX - dragState.startX;
        const dy = e.clientY - dragState.startY;
        if (!didDrag && (Math.abs(dx) > 4 || Math.abs(dy) > 4)) didDrag = true;
        if (didDrag) applyPosition(dragState.startLeft + dx, dragState.startTop + dy);
    }

    function stopDrag() {
        document.removeEventListener("pointermove", onDrag);
        if (didDrag) {
            const rect = overlay.getBoundingClientRect();
            chrome.storage.sync.set({ autobotPos: { left: rect.left, top: rect.top } });
            // Swallow the click that naturally follows mouseup after a real
            // drag, so releasing on the FAB doesn't also toggle the panel.
            const suppressClick = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
            fab.addEventListener("click", suppressClick, { capture: true, once: true });
        }
        dragState = null;
    }

    header.addEventListener("pointerdown", startDrag);
    fab.addEventListener("pointerdown", startDrag);

    // Keep it on-screen if the window gets resized after being dragged.
    window.addEventListener("resize", () => {
        if (overlay.style.left && overlay.style.top) {
            applyPosition(parseFloat(overlay.style.left), parseFloat(overlay.style.top));
        }
    });

    // ── State ────────────────────────────────────────────────────────────────
    let serverUrl = "http://127.0.0.1:8000";
    let wsUrl = "ws://127.0.0.1:8000";
    let ws = null;
    let isRunning = false;
    let showPeek = false;
    let chatHistory = []; // [{role: "user"|"assistant", content: string}] — sent to /api/chat for context
    let shownApprovalKey = null; // last human_input key we already rendered a card for
    let approvalPoll = null;

    // ── Load server URL from storage ─────────────────────────────────────────
    chrome.storage.sync.get(["autobotServerUrl", "autobotShowPeek", "autobotPos"], (result) => {
        if (result.autobotServerUrl) {
            serverUrl = result.autobotServerUrl.replace(/\/$/, "");
            wsUrl = serverUrl.replace(/^http/, "ws");
            chrome.runtime.sendMessage({ type: "SET_SERVER_URL", url: serverUrl });
        }
        if (result.autobotShowPeek !== undefined) {
            showPeek = result.autobotShowPeek;
            if (showPeek) miniPeek.classList.remove("hidden");
        }
        if (result.autobotPos && typeof result.autobotPos.left === "number") {
            applyPosition(result.autobotPos.left, result.autobotPos.top);
        }
        updateContext();
        startStatusPoll();
    });

    // ── Page context ─────────────────────────────────────────────────────────
    function updateContext() {
        const url = window.location.href;
        const title = document.title;
        if (ctxEl) ctxEl.textContent = `📄 ${title} — ${url.slice(0, 60)}${url.length > 60 ? "..." : ""}`;
    }

    // ── API helper (relays through background.js to dodge CORS) ────────────────
    function apiCall(path, method, body) {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage(
                { type: "AUTOBOT_API_CALL", url: `${serverUrl}${path}`, method, body },
                (resp) => resolve(resp || { ok: false, error: "No response from background script" })
            );
        });
    }

    // ── Chat feed helpers ────────────────────────────────────────────────────
    function clearEmptyState() {
        const placeholder = chatEl.querySelector(".autobot-chat-empty");
        if (placeholder) placeholder.remove();
    }

    function trimFeed() {
        while (chatEl.children.length > 400) chatEl.removeChild(chatEl.firstChild);
    }

    // role: "user" | "assistant" | "system"
    function appendBubble(role, text, opts = {}) {
        clearEmptyState();
        const bubble = document.createElement("div");
        bubble.className = `autobot-bubble autobot-bubble-${role}` + (opts.thinking ? " autobot-bubble-thinking" : "");
        bubble.textContent = text;
        chatEl.appendChild(bubble);
        chatEl.scrollTop = chatEl.scrollHeight;
        trimFeed();
        return bubble;
    }

    function appendPlanCard(plan) {
        clearEmptyState();
        const card = document.createElement("div");
        card.className = "autobot-plan-card";

        const nameEl = document.createElement("div");
        nameEl.className = "autobot-plan-name";
        nameEl.textContent = `📋 ${plan.name || "Task Plan"}`;
        card.appendChild(nameEl);

        if (plan.description) {
            const descEl = document.createElement("div");
            descEl.className = "autobot-plan-desc";
            descEl.textContent = plan.description;
            card.appendChild(descEl);
        }

        const stepsEl = document.createElement("ol");
        stepsEl.className = "autobot-plan-steps";
        for (const step of plan.steps || []) {
            const li = document.createElement("li");
            li.textContent = (step && step.description) || String(step);
            stepsEl.appendChild(li);
        }
        card.appendChild(stepsEl);

        const startBtn = document.createElement("button");
        startBtn.className = "autobot-btn autobot-plan-start";
        startBtn.textContent = "▶ Start this";
        startBtn.addEventListener("click", () => {
            startBtn.disabled = true;
            startBtn.textContent = "Starting…";
            startPlan(plan, startBtn);
        });
        card.appendChild(startBtn);

        chatEl.appendChild(card);
        chatEl.scrollTop = chatEl.scrollHeight;
        trimFeed();
    }

    function appendApprovalCard(key, message) {
        clearEmptyState();
        const card = document.createElement("div");
        card.className = "autobot-approval-card";

        const msgEl = document.createElement("div");
        msgEl.className = "autobot-approval-msg";
        msgEl.textContent = `🔒 ${message || "Autobot wants to do something that needs your approval."}`;
        card.appendChild(msgEl);

        const actions = document.createElement("div");
        actions.className = "autobot-approval-actions";
        const allowBtn = document.createElement("button");
        allowBtn.className = "autobot-btn autobot-approve-btn";
        allowBtn.textContent = "✓ Allow";
        const blockBtn = document.createElement("button");
        blockBtn.className = "autobot-btn autobot-block-btn";
        blockBtn.textContent = "✕ Block";
        actions.appendChild(allowBtn);
        actions.appendChild(blockBtn);
        card.appendChild(actions);

        const respond = async (response) => {
            allowBtn.disabled = true;
            blockBtn.disabled = true;
            const resp = await apiCall("/api/human_input/respond", "POST", { key, response });
            msgEl.textContent += resp && resp.ok
                ? (response === "allow" ? " — Allowed ✓" : " — Blocked ✕")
                : " — couldn't reach Autobot, try again";
            if (shownApprovalKey === key) shownApprovalKey = null;
        };
        allowBtn.addEventListener("click", () => respond("allow"));
        blockBtn.addEventListener("click", () => respond("block"));

        chatEl.appendChild(card);
        chatEl.scrollTop = chatEl.scrollHeight;
        trimFeed();
    }

    // Compact inline trace line for the raw execution log (WS stream) — kept
    // in the SAME feed as the chat so you see the conversation and what
    // Autobot is actually doing together, instead of two disconnected panels.
    function addTraceLine(text) {
        clearEmptyState();
        const line = document.createElement("div");
        let cls = "autobot-trace-line";
        if (text.includes("[PHASE") || text.includes("======")) cls += " autobot-trace-phase";
        else if (text.includes("[VERIFIER]")) cls += " autobot-trace-verifier";
        // 🏁/🏆 are CoreLoop's own final-result markers (AgentRunner.run()'s
        // "🏁 Finished: ..." and CoreLoop._handle_done()'s "🏆 Done: ...").
        else if (text.includes("✅") || text.includes("🏆") || text.includes("🏁") || text.includes("Goal achieved")) cls += " autobot-trace-success";
        else if (text.includes("❌") || text.includes("📛") || text.includes("ERROR") || text.includes("failed")) cls += " autobot-trace-error";
        else if (text.includes("⚠")) cls += " autobot-trace-warn";
        line.className = cls;
        line.textContent = text;
        chatEl.appendChild(line);
        chatEl.scrollTop = chatEl.scrollHeight;
        trimFeed();
    }

    // ── WebSocket log stream ──────────────────────────────────────────────────
    function connectWs() {
        if (ws) { try { ws.close(); } catch (_) { } }
        ws = new WebSocket(`${wsUrl}/ws/logs`);
        ws.onmessage = (e) => {
            if (e.data && e.data !== "__ping__") addTraceLine(e.data);
        };
        ws.onclose = () => {
            if (isRunning) setTimeout(connectWs, 2000);
        };
        ws.onerror = () => { ws = null; };
    }

    // ── Approval polling (only while a task is running) ─────────────────────
    function startApprovalPoll() {
        stopApprovalPoll();
        checkApproval();
        approvalPoll = setInterval(checkApproval, 2000);
    }
    function stopApprovalPoll() {
        if (approvalPoll) { clearInterval(approvalPoll); approvalPoll = null; }
    }
    async function checkApproval() {
        const resp = await apiCall("/api/human_input", "GET");
        if (!resp || !resp.ok) return;
        const data = resp.data || {};
        if (data.pending && data.key !== shownApprovalKey) {
            shownApprovalKey = data.key;
            appendApprovalCard(data.key, data.message);
        }
    }

    // ── Status polling (uses background sync) ─────────────────────────────────
    let statusPoll = null;
    function startStatusPoll() {
        stopStatusPoll();
        statusPoll = setInterval(() => {
            chrome.runtime.sendMessage({ type: "GET_AUTOBOT_STATUS" }, (status) => {
                if (!status) return;

                const s = status.status || "idle";
                if (statusEl) statusEl.textContent = s;

                if (s === "running" && !isRunning) {
                    setRunning(false); // sync local state without re-triggering API
                } else if (s !== "running" && isRunning) {
                    setIdle(s);
                }

                if (showPeek) {
                    if (status.browser_active) {
                        miniPeek.classList.add("has-image");
                        peekImg.src = `${serverUrl}/api/browser/screenshot?t=${Date.now()}`;
                    } else {
                        miniPeek.classList.remove("has-image");
                    }
                }
            });
        }, 3000);
    }

    function stopStatusPoll() {
        if (statusPoll) { clearInterval(statusPoll); statusPoll = null; }
    }

    // ── Run / Stop / Steer ──────────────────────────────────────────────────────
    function setRunning(triggerApi = true) {
        isRunning = true;
        if (runBanner) runBanner.classList.remove("hidden");
        if (runBannerText) runBannerText.textContent = triggerApi ? "⟳ Starting…" : "⟳ Running…";
        if (statusEl) statusEl.textContent = triggerApi ? "starting" : "running";
        connectWs();
        startApprovalPoll();
    }

    function setIdle(finalStatus) {
        const wasRunning = isRunning;
        isRunning = false;
        if (runBanner) runBanner.classList.add("hidden");
        if (ws) { try { ws.close(); } catch (_) { } ws = null; }
        stopApprovalPoll();
        shownApprovalKey = null;
        if (wasRunning && finalStatus) {
            appendBubble("system", finalStatus === "failed" ? "❌ Run finished — failed" : "✅ Run finished");
        }
    }

    async function startPlan(plan, startBtnEl) {
        const goal = [plan.description, ...(plan.steps || []).map((s) => (s && s.description) || String(s))]
            .filter(Boolean)
            .join("\n");
        const resp = await apiCall("/api/agent/run", "POST", { goal });
        if (!resp || !resp.ok) {
            if (startBtnEl) { startBtnEl.disabled = false; startBtnEl.textContent = "▶ Start this"; }
            const detail = (resp && resp.data && resp.data.detail) || (resp && resp.error);
            appendBubble("system", resp && resp.status === 409
                ? "⚠ A task is already running — wait for it to finish, or hit Stop first."
                : `❌ Couldn't start: ${detail || "unknown error"}`);
            return;
        }
        if (startBtnEl) startBtnEl.textContent = "▶ Started";
        setRunning(true);
        appendBubble("system", `🚀 Started: ${plan.name || "task"}`);
    }

    async function sendMessage() {
        const text = (goalInput?.value || "").trim();
        if (!text) return;
        goalInput.value = "";
        autoGrow();
        appendBubble("user", text);

        if (isRunning) {
            // Steer the already-running agent instead of doing nothing / erroring.
            const resp = await apiCall("/api/agent/message", "POST", { text });
            if (!resp || !resp.ok) {
                appendBubble("system", `❌ Couldn't reach the running agent: ${(resp && resp.data && resp.data.detail) || (resp && resp.error) || "unknown error"}`);
            }
            // Confirmation ("⚡ Mid-flight pivot pushed…") arrives via the WS trace stream.
            return;
        }

        chatHistory.push({ role: "user", content: text });
        const thinking = appendBubble("assistant", "…", { thinking: true });
        const resp = await apiCall("/api/chat", "POST", { message: text, history: chatHistory.slice(-10) });
        thinking.remove();

        if (!resp || !resp.ok) {
            appendBubble("system", `❌ Couldn't reach Autobot: ${(resp && resp.data && resp.data.detail) || (resp && resp.error) || "unknown error"}`);
            return;
        }
        const data = resp.data || {};
        const reply = data.reply || "…";
        appendBubble("assistant", reply);
        chatHistory.push({ role: "assistant", content: reply });
        if (data.plan) appendPlanCard(data.plan);
    }

    async function stopGoal() {
        const resp = await apiCall("/api/agent/cancel", "POST");
        appendBubble("system", resp && resp.ok ? "⚠ Stop requested." : "⚠ Stop requested (backend didn't confirm).");
    }

    // ── Input auto-grow (chat-style: grows up to ~5 lines, then scrolls) ───────
    function autoGrow() {
        if (!goalInput) return;
        goalInput.style.height = "auto";
        goalInput.style.height = Math.min(goalInput.scrollHeight, 110) + "px";
    }

    // ── UI Events ─────────────────────────────────────────────────────────────
    fab?.addEventListener("click", () => {
        panel?.classList.toggle("open");
        updateContext();
    });
    closeBtn?.addEventListener("click", () => {
        panel?.classList.remove("open");
    });
    runBtn?.addEventListener("click", sendMessage);
    stopBtn?.addEventListener("click", stopGoal);

    peekToggle?.addEventListener("click", () => {
        showPeek = !showPeek;
        chrome.storage.sync.set({ autobotShowPeek: showPeek });
        miniPeek.classList.toggle("hidden", !showPeek);
    });

    goalInput?.addEventListener("input", autoGrow);
    goalInput?.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

})();
