/**
 * Autobot Extension — Content Script
 * Injected into every page. Creates a floating overlay chatbot that:
 *  - Shows the current page context (URL + title)
 *  - Lets the user type a goal and send it to the local Autobot backend
 *  - Streams live log output via WebSocket
 */

(function () {
    "use strict";

    // Avoid injecting twice
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
        <button id="autobot-close-btn" title="Minimize">✕</button>
      </div>
      <div id="autobot-page-context">📄 Loading page context...</div>
      <div id="autobot-logs"><span class="autobot-log-empty">Logs will appear here...</span></div>
      <div id="autobot-input-area">
        <textarea id="autobot-goal-input" placeholder="Tell Autobot what to do on this page...&#10;e.g. &quot;Enter this Kaggle competition and aim for 80% accuracy&quot;" rows="3"></textarea>
        <div id="autobot-actions">
          <button id="autobot-run-btn" class="autobot-btn">▶ Run Goal</button>
          <button id="autobot-stop-btn" class="autobot-btn" disabled>■ Stop</button>
        </div>
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
    const logsEl = document.getElementById("autobot-logs");
    const statusEl = document.getElementById("autobot-header-status");
    const ctxEl = document.getElementById("autobot-page-context");
    const goalInput = document.getElementById("autobot-goal-input");
    const runBtn = document.getElementById("autobot-run-btn");
    const stopBtn = document.getElementById("autobot-stop-btn");

    // ── State ────────────────────────────────────────────────────────────────
    let serverUrl = "http://127.0.0.1:8000";
    let wsUrl = "ws://127.0.0.1:8000";
    let ws = null;
    let isRunning = false;

    // ── Load server URL from storage ─────────────────────────────────────────
    chrome.storage.sync.get(["autobotServerUrl"], (result) => {
        if (result.autobotServerUrl) {
            serverUrl = result.autobotServerUrl.replace(/\/$/, "");
            wsUrl = serverUrl.replace(/^http/, "ws");
        }
        updateContext();
    });

    // ── Page context ─────────────────────────────────────────────────────────
    function updateContext() {
        const url = window.location.href;
        const title = document.title;
        if (ctxEl) ctxEl.textContent = `📄 ${title} — ${url.slice(0, 60)}${url.length > 60 ? "..." : ""}`;
    }

    function getPageText() {
        try {
            // Grab visible body text, limit to 8000 chars to avoid huge payloads
            const el = document.body || document.documentElement;
            return el.innerText.slice(0, 8000);
        } catch (_) {
            return "";
        }
    }

    // ── Log helpers ───────────────────────────────────────────────────────────
    function addLog(text) {
        if (!logsEl) return;
        // Clear placeholder
        const placeholder = logsEl.querySelector(".autobot-log-empty");
        if (placeholder) placeholder.remove();

        const line = document.createElement("div");
        let cls = "autobot-log-line";
        if (text.includes("[PHASE") || text.includes("======")) cls = "autobot-log-phase";
        else if (text.includes("[VERIFIER]")) cls = "autobot-log-verifier";
        else if (text.includes("✅") || text.includes("Goal achieved")) cls = "autobot-log-success";
        else if (text.includes("❌") || text.includes("ERROR") || text.includes("failed")) cls = "autobot-log-error";
        else if (text.includes("⚠")) cls = "autobot-log-warn";
        line.className = cls;
        line.textContent = text;
        logsEl.appendChild(line);
        logsEl.scrollTop = logsEl.scrollHeight;
    }

    function clearLogs() {
        if (logsEl) logsEl.innerHTML = '<span class="autobot-log-empty">Starting...</span>';
    }

    // ── WebSocket log stream ──────────────────────────────────────────────────
    function connectWs() {
        if (ws) { try { ws.close(); } catch (_) { } }
        ws = new WebSocket(`${wsUrl}/ws/logs`);
        ws.onmessage = (e) => {
            if (e.data && e.data !== "__ping__") addLog(e.data);
        };
        ws.onclose = () => {
            if (isRunning) setTimeout(connectWs, 2000); // reconnect while running
        };
        ws.onerror = () => { ws = null; };
    }

    // ── Status polling ────────────────────────────────────────────────────────
    let statusPoll = null;
    function startStatusPoll() {
        stopStatusPoll();
        statusPoll = setInterval(async () => {
            try {
                const r = await fetch(`${serverUrl}/api/autonomous/status`);
                const data = await r.json();
                const s = data.status || "idle";
                if (statusEl) statusEl.textContent = s === "running"
                    ? `⟳ Phase ${(data.current_phase_index ?? 0) + 1}/${(data.phase_plan || []).length || "?"}`
                    : s;
                if (s !== "running") {
                    setIdle();
                }
            } catch (_) { }
        }, 2000);
    }

    function stopStatusPoll() {
        if (statusPoll) { clearInterval(statusPoll); statusPoll = null; }
    }

    // ── Run / Stop ────────────────────────────────────────────────────────────
    function setRunning() {
        isRunning = true;
        if (runBtn) runBtn.disabled = true;
        if (stopBtn) stopBtn.disabled = false;
        if (goalInput) goalInput.disabled = true;
        if (statusEl) statusEl.textContent = "⟳ Starting...";
        connectWs();
        startStatusPoll();
    }

    function setIdle() {
        isRunning = false;
        if (runBtn) runBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        if (goalInput) goalInput.disabled = false;
        stopStatusPoll();
        if (ws) { try { ws.close(); } catch (_) { } ws = null; }
    }

    async function runGoal() {
        const goal = (goalInput?.value || "").trim();
        if (!goal) {
            if (goalInput) goalInput.placeholder = "Please enter a goal first!";
            return;
        }
        clearLogs();
        setRunning();
        addLog(`🤖 Goal: ${goal}`);

        const payload = {
            goal,
            page_context: {
                url: window.location.href,
                title: document.title,
                text: getPageText(),
            },
            max_hours: 8.0,
        };

        try {
            const resp = await fetch(`${serverUrl}/api/run_autonomous`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const err = await resp.json();
                addLog(`❌ Error: ${err.detail || resp.statusText}`);
                setIdle();
                return;
            }
            const data = await resp.json();
            addLog(`✓ Run started: ${data.run_id}`);
        } catch (e) {
            addLog(`❌ Cannot reach Autobot at ${serverUrl}. Is it running?`);
            setIdle();
        }
    }

    async function stopGoal() {
        try {
            await fetch(`${serverUrl}/api/autonomous/cancel`, { method: "POST" });
            addLog("⚠ Stop requested.");
        } catch (e) {
            addLog(`❌ Could not cancel: ${e.message}`);
        }
        setIdle();
    }

    // ── Panel toggle ──────────────────────────────────────────────────────────
    fab?.addEventListener("click", () => {
        panel?.classList.toggle("open");
        updateContext();
        if (!isRunning) {
            // Ping backend to warm ws connection
            fetch(`${serverUrl}/api/status`).catch(() => { });
        }
    });
    closeBtn?.addEventListener("click", () => {
        panel?.classList.remove("open");
    });
    runBtn?.addEventListener("click", runGoal);
    stopBtn?.addEventListener("click", stopGoal);

    // Ctrl+Enter submits
    goalInput?.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && e.ctrlKey) runGoal();
    });

})();
