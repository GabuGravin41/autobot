"""Browser smoke test -- connects to Chrome on port 9222 and performs a web task."""
import httpx
import time
import sys

# Force UTF-8 output for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"

print("=" * 50)
print("Autobot Live Browser Test")
print("=" * 50)

# -- 1. Check server is up ---------------------------------
try:
    ping = httpx.get(f"{BASE}/api/status", timeout=5)
    print(f"[OK] Server responding ({ping.status_code})")
except Exception as e:
    print(f"[FAIL] Server not reachable: {e}")
    print("      Run first: python -m autobot.cli --server")
    sys.exit(1)

# -- 1b. Cancel any existing task --------------------------
status_now = httpx.get(f"{BASE}/api/status", timeout=5).json().get("status", "idle")
if status_now in ("running", "starting"):
    print(f"[..] Cancelling previous task ({status_now})...")
    httpx.post(f"{BASE}/api/agent/cancel", timeout=5)
    time.sleep(2)

# -- 2. Submit a browser goal ------------------------------
goal = "Navigate to https://example.com and tell me the heading text on the page."
r = httpx.post(
    f"{BASE}/api/agent/run",
    json={"goal": goal, "max_steps": 6},
    timeout=10,
)

data = r.json()
run_id = data.get("run_id") or data.get("status", "?")
print(f"[OK] Browser task submitted: {run_id} (HTTP {r.status_code})")
print("      Polling every 3s...")
print()

# -- 3. Poll for result (2 min max) ------------------------
final = {}
for i in range(40):          # 40 x 3s = 2 minutes
    time.sleep(3)
    s = httpx.get(f"{BASE}/api/status", timeout=5).json()
    status = s.get("status", "?")
    step   = s.get("current_step", 0)
    total  = s.get("max_steps", 6)
    elapsed = (i + 1) * 3
    print(f"  [{elapsed:>3}s] status={status:<12} step={step}/{total}")
    final = s
    if status in ("done", "completed", "failed", "cancelled"):
        break

print()
print("-" * 50)
print("Final status :", final.get("status"))
print("Result       :", (final.get("result") or "(empty)")[:500])
if final.get("history"):
    print()
    print("Step History:")
    for h in final["history"]:
        print(" ", h[:200])
