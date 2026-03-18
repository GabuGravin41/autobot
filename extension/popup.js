/**
 * Autobot Extension — popup.js
 *
 * Live dashboard in the extension popup:
 *  - Polls /api/status every 2s for run state, narrative, screenshot
 *  - Shows Allow/Block buttons when approval is pending
 *  - Abort run button while a task is running
 *  - Settings section to change backend URL
 */

const $ = (id) => document.getElementById(id);

const els = {
    dot:         $("status-dot"),
    screenshot:  $("screenshot"),
    ssPlaceholder: $("screenshot-placeholder"),
    badge:       $("run-badge"),
    stepInfo:    $("step-info"),
    modelInfo:   $("model-info"),
    narrativeBox:$("narrative-box"),
    goalBox:     $("goal-box"),
    approvalBox: $("approval-box"),
    approvalMsg: $("approval-msg"),
    allowBtn:    $("allow-btn"),
    blockBtn:    $("block-btn"),
    controls:    $("controls"),
    abortBtn:    $("abort-btn"),
    dashLink:    $("dashboard-link"),
    urlInput:    $("server-url"),
    saveBtn:     $("save-btn"),
    savedMsg:    $("saved-msg"),
};

let serverUrl = "http://127.0.0.1:8000";
let pollTimer = null;
let currentApprovalKey = null;

// ── Init ──────────────────────────────────────────────────────────────────────

chrome.storage.sync.get(["autobotServerUrl"], (r) => {
    serverUrl = (r.autobotServerUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
    els.urlInput.value = serverUrl;
    els.dashLink.href = serverUrl;
    startPolling();
});

// ── Polling ───────────────────────────────────────────────────────────────────

function startPolling() {
    poll();
    pollTimer = setInterval(poll, 2000);
}

async function poll() {
    try {
        const [statusRes, approvalRes] = await Promise.all([
            fetch(`${serverUrl}/api/status`, { signal: AbortSignal.timeout(3000) }),
            fetch(`${serverUrl}/api/human_input`, { signal: AbortSignal.timeout(3000) }),
        ]);

        if (!statusRes.ok) throw new Error(`HTTP ${statusRes.status}`);
        const status = await statusRes.json();
        const approval = approvalRes.ok ? await approvalRes.json() : { pending: false };

        renderStatus(status);
        renderApproval(approval);
        setDot("green");
    } catch (e) {
        setDot("red");
        els.badge.textContent = "OFFLINE";
        els.badge.className = "badge badge-failed";
        els.stepInfo.textContent = e.message;
        els.narrativeBox.style.display = "none";
        els.goalBox.style.display = "none";
        els.approvalBox.style.display = "none";
        els.controls.style.display = "none";
    }
}

function renderStatus(s) {
    // Badge
    const runStatus = s.run_status || "idle";
    els.badge.textContent = runStatus.toUpperCase();
    els.badge.className = `badge badge-${runStatus}`;

    // Step info
    const step = s.active_run_id ? (s.current_step || "") : "";
    const maxStep = s.max_steps || "";
    els.stepInfo.textContent = step ? `Step ${step}${maxStep ? "/" + maxStep : ""}` : "";

    // Model
    els.modelInfo.textContent = s.llm_model ? `${s.llm_provider || ""} / ${s.llm_model}` : "";

    // Screenshot
    if (s.screenshot_b64) {
        els.screenshot.src = `data:image/jpeg;base64,${s.screenshot_b64}`;
        els.screenshot.style.display = "block";
        els.ssPlaceholder.style.display = "none";
    } else {
        els.screenshot.style.display = "none";
        els.ssPlaceholder.style.display = "block";
    }

    // Narrative
    if (s.narrative) {
        els.narrativeBox.textContent = `💬 ${s.narrative}`;
        els.narrativeBox.style.display = "block";
    } else {
        els.narrativeBox.style.display = "none";
    }

    // Goal
    if (s.browser && s.browser.url && s.browser.url !== "about:blank") {
        els.goalBox.textContent = `🌐 ${s.browser.url}`;
        els.goalBox.style.display = "block";
    } else {
        els.goalBox.style.display = "none";
    }

    // Controls (abort) — show when running
    if (runStatus === "running") {
        els.controls.style.display = "flex";
        setDot("pulse");
    } else {
        els.controls.style.display = "none";
        setDot("green");
    }
}

function renderApproval(a) {
    if (a.pending && a.key && a.key !== currentApprovalKey) {
        currentApprovalKey = a.key;
        els.approvalMsg.textContent = a.message || "Agent is requesting permission to proceed.";
        els.approvalBox.style.display = "block";
    } else if (!a.pending) {
        currentApprovalKey = null;
        els.approvalBox.style.display = "none";
    }
}

// ── Approval buttons ──────────────────────────────────────────────────────────

els.allowBtn.addEventListener("click", () => submitApproval("allow"));
els.blockBtn.addEventListener("click", () => submitApproval("block"));

async function submitApproval(response) {
    if (!currentApprovalKey) return;
    try {
        await fetch(`${serverUrl}/api/human_input`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ key: currentApprovalKey, response }),
        });
        currentApprovalKey = null;
        els.approvalBox.style.display = "none";
    } catch (e) {
        console.error("Approval submit failed:", e);
    }
}

// ── Abort ─────────────────────────────────────────────────────────────────────

els.abortBtn.addEventListener("click", async () => {
    els.abortBtn.disabled = true;
    els.abortBtn.textContent = "Aborting…";
    try {
        await fetch(`${serverUrl}/api/agent/cancel`, { method: "POST" });
    } catch {}
    setTimeout(() => {
        els.abortBtn.disabled = false;
        els.abortBtn.textContent = "Abort Run";
    }, 3000);
});

// ── Settings ──────────────────────────────────────────────────────────────────

els.saveBtn.addEventListener("click", () => {
    const url = (els.urlInput.value || "http://127.0.0.1:8000").replace(/\/$/, "");
    chrome.storage.sync.set({ autobotServerUrl: url }, () => {
        serverUrl = url;
        els.dashLink.href = url;
        els.savedMsg.textContent = "✓ Saved";
        setTimeout(() => { els.savedMsg.textContent = ""; }, 2000);
        poll();  // immediate re-check
    });
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function setDot(state) {
    els.dot.className = `dot ${state}`;
}
