"""
demo_extension_bridge.py — Prove the CDP-free browser bridge actually works.

This talks to your running Autobot backend's /api/extension/* endpoints,
which relay to the Autobot Chrome extension in your real, already-logged-in
Chrome (see autobot/browser/extension_bridge.py for how the round trip
works, and CoreLoop's browser_text/browser_list/browser_click/browser_type
actions in autobot/agent/core_loop.py for how the agent uses the same
bridge).

No CDP, no separate browser profile, no debug port — this drives whatever
tab is currently focused in your normal Chrome, through the extension's
already-granted content-script permissions.

Setup (one time):
  1. chrome://extensions → enable Developer mode → "Load unpacked" → select
     the `extension/` folder (or, if already loaded, click the reload icon
     to pick up this update).
  2. Click the Autobot toolbar icon once on any normal http(s) page — this
     confirms the extension's content script is alive on that tab.
  3. Start the backend:  autobot --server   (or: python -m autobot.main)

Usage:
  python demo_extension_bridge.py                 # status + read + list
  python demo_extension_bridge.py read             # just read the page text
  python demo_extension_bridge.py list             # list clickable elements, numbered
  python demo_extension_bridge.py click 3          # click element [3] from the last `list`
  python demo_extension_bridge.py type 2 "hello"   # type "hello" into element [2]

Before "click"/"type": run "list" first on the SAME page state — indices
are recomputed fresh every call, so a stale index from an earlier page
state may point at the wrong element (or fail with a clear error, never
silently misclick — same rule the rest of Autobot's approval-gated actions
follow).
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BACKEND = "http://127.0.0.1:8000"


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BACKEND}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BACKEND}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_backend() -> bool:
    try:
        _get("/api/health")
        return True
    except Exception as e:
        print(f"✗ Can't reach the Autobot backend at {BACKEND}: {e}")
        print("  Start it first:  autobot --server   (or: python -m autobot.main)")
        return False


def check_extension() -> bool:
    try:
        status = _get("/api/extension/status")
    except Exception as e:
        return False  # already reported by check_backend
    connected = status.get("connected", False)
    if connected:
        print("✓ Extension is connected (polled the backend within the last 10s).")
    else:
        print("✗ Extension has not polled the backend recently. Checklist:")
        print("  1. chrome://extensions → Autobot is loaded and enabled")
        print("     (if you just updated the code, click the reload icon there)")
        print("  2. The extension's server URL matches this backend (click the")
        print(f"     toolbar icon → should say {BACKEND})")
        print("  3. Chrome itself is open (the background service worker only")
        print("     runs while Chrome is running)")
    return connected


def run_command(cmd_type: str, timeout: float = 10.0, **params) -> None:
    print(f"→ Sending '{cmd_type}' {params or ''} to whatever tab is focused in your Chrome...")
    result = _post("/api/extension/command", {"type": cmd_type, "params": params, "timeout": timeout})
    if result.get("ok"):
        print(f"✓ {cmd_type} succeeded:")
        print(json.dumps(result.get("data"), indent=2, ensure_ascii=False)[:3000])
    else:
        print(f"✗ {cmd_type} failed: {result.get('error')}")


def main() -> None:
    if not check_backend():
        sys.exit(1)

    args = sys.argv[1:]
    action = args[0] if args else "smoke"

    if action == "status":
        check_extension()
        return

    if not check_extension():
        sys.exit(1)

    if action == "smoke" or action == "read":
        run_command("read_text")
        if action == "smoke":
            print()
            run_command("list_elements")
        return

    if action == "list":
        run_command("list_elements")
        return

    if action == "click":
        if len(args) < 2:
            print("Usage: python demo_extension_bridge.py click <index>")
            sys.exit(1)
        run_command("click_index", index=int(args[1]))
        return

    if action == "type":
        if len(args) < 3:
            print('Usage: python demo_extension_bridge.py type <index> "text"')
            sys.exit(1)
        run_command("type_index", index=int(args[1]), text=" ".join(args[2:]))
        return

    print(f"Unknown action: {action!r}. See the module docstring for usage.")
    sys.exit(1)


if __name__ == "__main__":
    main()
