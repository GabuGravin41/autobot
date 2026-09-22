"""
demo_agent_run.py — Run a real goal through CoreLoop, driven by your
actually-configured LLM (whatever AUTOBOT_LLM_PROVIDER/AUTOBOT_LLM_MODEL
are set to in .env), and watch it decide, step by step, whether to
navigate and whether to call the browser_* extension-bridge actions.

This is different from demo_extension_bridge.py: that script called the
bridge directly over HTTP, proving the *mechanism* works. This script
proves the *agent* can use it — the LLM reads system_prompt.md, decides
which action to take from real screen state, and the loop executes it.

Zero extra dependencies (stdlib only), same as demo_extension_bridge.py.

Usage:
  python demo_agent_run.py                      # runs the built-in Overleaf goal
  python demo_agent_run.py "your own goal here"  # runs any goal you type

Before running: make sure `autobot --server` (or `python -m autobot.main`)
is running, the Autobot Chrome extension is loaded/enabled, and a normal
Chrome tab (not chrome://extensions) is the active one.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BACKEND = "http://127.0.0.1:8000"

DEFAULT_GOAL = (
    "Check the current browser tab. If it is NOT already Overleaf's project "
    "list page (https://www.overleaf.com/project), use the navigate action "
    "to open that URL. Once you are there, use browser_list or browser_text "
    "to read the list of projects, identify the project at the very top of "
    "the list (the most recently modified one), and call done with its "
    "exact name in the text field. This is a read-only task — do not click "
    "anything on the page."
)


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BACKEND}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{BACKEND}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"detail": body}


def main() -> None:
    goal = " ".join(sys.argv[1:]).strip() or DEFAULT_GOAL

    try:
        _get("/api/health")
    except Exception as e:
        print(f"✗ Can't reach the Autobot backend at {BACKEND}: {e}")
        print("  Start it first:  autobot --server   (or: python -m autobot.main)")
        sys.exit(1)

    ext_status = _get("/api/extension/status")
    if not ext_status.get("connected"):
        print("✗ Extension not connected — browser_* actions won't work.")
        print("  chrome://extensions → reload Autobot, then keep a normal tab focused.")
        sys.exit(1)
    print("✓ Backend reachable, extension connected.\n")

    print(f"Goal: {goal}\n")
    status, resp = _post("/api/agent/run", {"goal": goal, "max_steps": 12})
    if status == 409:
        print(f"✗ {resp.get('detail', 'A run is already in progress.')}")
        print("  Wait for it to finish, or restart the backend, then try again.")
        sys.exit(1)
    if status != 200:
        print(f"✗ Could not start run (HTTP {status}): {resp}")
        sys.exit(1)

    run_id = resp.get("run_id")
    print(f"→ Run started: {run_id}\n{'-'*70}")

    seen = 0
    deadline = time.time() + 180  # 3 minutes — generous for a 12-step run on a cheap model
    last_status = "running"
    while time.time() < deadline:
        try:
            logs = _get("/api/logs?limit=500").get("logs", [])
        except Exception:
            logs = []
        for line in logs[seen:]:
            print(line)
        seen = len(logs)

        try:
            st = _get("/api/agent/status")
        except Exception:
            st = {}
        last_status = st.get("agent_status") or st.get("run_status") or "running"
        if last_status in ("done", "failed", "cancelled"):
            break
        time.sleep(1.0)

    print(f"{'-'*70}\nFinal status: {last_status}")
    if last_status == "running":
        print("(Still running after 3 minutes — check the dashboard or /api/logs directly.)")


if __name__ == "__main__":
    main()
