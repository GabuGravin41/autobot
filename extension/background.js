/**
 * Autobot Extension — Background Service Worker
 * - Handles API relays (CORS bypass)
 * - Persists agent status across all tabs
 * - Polls backend for updates
 * - DOM command bridge: polls the backend for pending browser_* commands
 *   and runs them in the active tab's content script. This is what lets
 *   the Python backend read/click/type on a real page without CDP — see
 *   autobot/browser/extension_bridge.py for the Python side of this.
 */

let autobotStatus = {
    status: "idle",
    current_run_id: null,
    last_update: 0
};

let serverUrl = "http://127.0.0.1:8000";

// Periodically poll the backend for the global agent status
async function pollStatus() {
    try {
        const r = await fetch(`${serverUrl}/api/status`);
        if (r.ok) {
            const data = await r.json();
            autobotStatus = {
                status: data.run_status || "idle",
                active_run_id: data.active_run_id,
                browser_active: data.browser?.active || false,
                last_update: Date.now()
            };
        }
    } catch (e) {
        autobotStatus.status = "offline";
    }
}

setInterval(pollStatus, 3000);

// ── DOM command bridge ───────────────────────────────────────────────────
//
// Polls GET /api/extension/poll. When a command is waiting, runs it in the
// active tab via the content script (injecting the content script first if
// it isn't there yet — e.g. a tab opened before the extension last
// reloaded), then POSTs the result to /api/extension/result.

async function postCommandResult(id, ok, data, error) {
    try {
        await fetch(`${serverUrl}/api/extension/result`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id, ok, data, error: error || "" }),
        });
    } catch (e) {
        // Backend went away mid-command — nothing to do, the command's
        // Python-side caller will simply time out.
    }
}

function sendToTab(tabId, command) {
    return new Promise((resolve) => {
        chrome.tabs.sendMessage(tabId, { type: "AUTOBOT_RUN_DOM_COMMAND", command }, (response) => {
            resolve({ error: chrome.runtime.lastError, response });
        });
    });
}

async function runCommandInActiveTab(command) {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (!tab || !tab.id) {
        await postCommandResult(command.id, false, null, "No active tab found.");
        return;
    }

    let { error, response } = await sendToTab(tab.id, command);

    if (error) {
        // Content script probably isn't injected in this tab yet (e.g. the
        // tab was open before the extension was loaded/reloaded). Try
        // injecting it once, then retry.
        try {
            await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
            ({ error, response } = await sendToTab(tab.id, command));
        } catch (injErr) {
            await postCommandResult(
                command.id, false, null,
                `Could not reach the page (${injErr.message}). This won't work on chrome:// pages, ` +
                `the Chrome Web Store, or the extensions page itself — switch to a normal tab.`
            );
            return;
        }
    }

    if (error) {
        await postCommandResult(command.id, false, null, `Content script did not respond: ${error.message}`);
        return;
    }
    if (!response) {
        await postCommandResult(command.id, false, null, "Content script gave no response.");
        return;
    }
    await postCommandResult(command.id, response.ok, response.data, response.error || "");
}

async function pollCommands() {
    try {
        const r = await fetch(`${serverUrl}/api/extension/poll`);
        if (!r.ok) return;
        const data = await r.json();
        if (data.command) await runCommandInActiveTab(data.command);
    } catch (e) {
        // Backend offline — the next poll will just try again.
    }
}

setInterval(pollCommands, 1200);

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "AUTOBOT_API_CALL") {
        const { url, method, body } = message;
        fetch(url, {
            method: method || "GET",
            headers: { "Content-Type": "application/json" },
            body: body ? JSON.stringify(body) : undefined,
        })
            .then((r) => r.json().then((data) => ({ ok: r.ok, status: r.status, data })))
            .then(sendResponse)
            .catch((err) => sendResponse({ ok: false, error: String(err) }));
        return true;
    }

    if (message.type === "GET_AUTOBOT_STATUS") {
        sendResponse(autobotStatus);
        return true;
    }

    if (message.type === "SET_SERVER_URL") {
        serverUrl = message.url.replace(/\/$/, "");
        pollStatus();
        sendResponse({ ok: true });
        return true;
    }
});
