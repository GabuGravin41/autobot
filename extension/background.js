/**
 * Autobot Extension — Background Service Worker
 * Handles messages from content scripts that need to relay API calls
 * to the local Autobot backend (bypasses content script CORS restrictions).
 */

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
        return true; // Keep message channel open for async response
    }
});
