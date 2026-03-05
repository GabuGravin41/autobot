const urlInput = document.getElementById("server-url");
const saveBtn = document.getElementById("save-btn");
const savedMsg = document.getElementById("saved-msg");
const statusDot = document.getElementById("status-dot");
const statusTxt = document.getElementById("status-text");

// Load saved URL
chrome.storage.sync.get(["autobotServerUrl"], (result) => {
    const url = result.autobotServerUrl || "http://127.0.0.1:8000";
    urlInput.value = url;
    checkConnection(url);
});

async function checkConnection(url) {
    statusDot.className = "";
    statusTxt.textContent = "Checking...";
    try {
        const r = await fetch(`${url.replace(/\/$/, "")}/api/status`, { signal: AbortSignal.timeout(3000) });
        if (r.ok) {
            const d = await r.json();
            statusDot.className = "green";
            statusTxt.textContent = `Connected — LLM: ${d.llm_model || "unknown"}`;
        } else {
            throw new Error(`HTTP ${r.status}`);
        }
    } catch (e) {
        statusDot.className = "red";
        statusTxt.textContent = `Offline: ${e.message}`;
    }
}

saveBtn.addEventListener("click", () => {
    const url = (urlInput.value || "http://127.0.0.1:8000").replace(/\/$/, "");
    chrome.storage.sync.set({ autobotServerUrl: url }, () => {
        savedMsg.textContent = "✓ Saved!";
        setTimeout(() => { savedMsg.textContent = ""; }, 2000);
        checkConnection(url);
    });
});
