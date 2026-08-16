"""Quick end-to-end smoke test -- no browser needed for the task itself."""
import httpx
import time
import sys

# Force UTF-8 output so box-drawing / emoji don't crash on Windows CP1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"

print("=" * 50)
print("Autobot Live Test")
print("=" * 50)

# -- 1. Check server is up ---------------------------------
try:
    ping = httpx.get(f"{BASE}/api/status", timeout=5)
    print(f"[OK] Server responding ({ping.status_code})")
except Exception as e:
    print(f"[FAIL] Server not reachable: {e}")
    print("      Run first:  python -m autobot.cli --server")
    sys.exit(1)

# -- 1b. Cancel any already-running task -------------------
status_now = httpx.get(f"{BASE}/api/status", timeout=5).json().get("status", "idle")
if status_now in ("running", "starting"):
    print(f"[..] Previous task still {status_now} -- cancelling it first...")
    httpx.post(f"{BASE}/api/agent/cancel", timeout=5)
    time.sleep(3)

# -- 2. Submit a simple run_command task -------------------
goal = "Use run_command to run this exact command: python -c \"print('RESULT:', 2+2)\" -- then call done(success=True) with the output."
r = httpx.post(
    f"{BASE}/api/agent/run",
    json={"goal": goal, "max_steps": 8},
    timeout=10,
)
if r.status_code == 409:
    print("[WARN] Server busy (409). Waiting 5s and retrying...")
    time.sleep(5)
    r = httpx.post(f"{BASE}/api/agent/run", json={"goal": goal, "max_steps": 8}, timeout=10)

data = r.json()
run_id = data.get("run_id") or data.get("status", "?")
print(f"[OK] Task submitted: {run_id}  (HTTP {r.status_code})")
print()
print("NOTE: First run may take 60-90s while Chrome launches.")
print("      Polling every 5s for up to 3 minutes...")
print()

# -- 3. Poll for result (3 min max) -------------------------
final = {}
for i in range(36):          # 36 x 5s = 3 minutes
    time.sleep(5)
    s = httpx.get(f"{BASE}/api/status", timeout=5).json()
    status = s.get("status", "?")
    step   = s.get("current_step", 0)
    total  = s.get("max_steps", 8)
    elapsed = (i + 1) * 5
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
    print("Last steps (with output):")
    for h in final["history"]:
        print(" ", h[:200])
