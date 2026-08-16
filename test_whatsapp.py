"""WhatsApp Web Test Script — Navigates to WhatsApp Web, searches for Dalton Omondi, and sends a test message."""
import httpx
import time
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"

print("=" * 60)
print("Autobot WhatsApp Web Integration Test")
print("=" * 60)

# 1. Check server status
try:
    ping = httpx.get(f"{BASE}/api/status", timeout=5)
    print(f"[OK] Server responding ({ping.status_code})")
except Exception as e:
    print(f"[FAIL] Server not reachable: {e}")
    print("      Run first: python -m autobot.cli --server")
    sys.exit(1)

# 2. Cancel any running task
status_now = httpx.get(f"{BASE}/api/status", timeout=5).json().get("status", "idle")
if status_now in ("running", "starting"):
    print(f"[..] Cancelling previous task ({status_now})...")
    httpx.post(f"{BASE}/api/agent/cancel", timeout=5)
    time.sleep(2)

# 3. Submit WhatsApp task
goal = (
    "Navigate to https://web.whatsapp.com. "
    "If the page shows a QR code, wait for the chat interface to load after the user scans it. "
    "Once the chat list is loaded, search for 'Dalton Omondi' in the search box (or click the search box), "
    "select the contact 'Dalton Omondi', and send the message: "
    "'Hello Dalton! Autobot WhatsApp integration test successful.' "
    "Once sent, call done(success=True)."
)

r = httpx.post(
    f"{BASE}/api/agent/run",
    json={"goal": goal, "max_steps": 20},
    timeout=30,
)

data = r.json()
run_id = data.get("run_id") or data.get("status", "?")
print(f"[OK] WhatsApp task submitted: {run_id} (HTTP {r.status_code})")
print("      Polling step progress every 4 seconds...")
print()

# 4. Poll progress
final = {}
for i in range(45):          # 45 x 4s = 3 minutes max
    time.sleep(4)
    s = httpx.get(f"{BASE}/api/status", timeout=5).json()
    status = s.get("status", "?")
    step   = s.get("current_step", 0)
    total  = s.get("max_steps", 12)
    elapsed = (i + 1) * 4
    print(f"  [{elapsed:>3}s] status={status:<12} step={step}/{total}")
    final = s
    if status in ("done", "completed", "failed", "cancelled"):
        break

print()
print("-" * 60)
print("Final status :", final.get("status"))
print("Result       :", (final.get("result") or "(empty)")[:500])
if final.get("history"):
    print()
    print("Step History:")
    for h in final["history"]:
        print(" ", h[:250])
