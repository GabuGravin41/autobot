/**
 * Autobot Extension — Background Service Worker
 * - Handles API relays (CORS bypass)
 * - Persists agent status across all tabs
 * - Polls backend for updates
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
