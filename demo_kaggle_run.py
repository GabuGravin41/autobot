"""
demo_kaggle_run.py — Run a real Kaggle kernel-iteration goal through
CoreLoop, driven by your actually-configured LLM, and watch it pick the
right computer.kaggle.* tool from the catalog on its own.

Unlike demo_agent_run.py (Overleaf, browser_* actions), this exercises a
different part of the tool catalog entirely: computer.kaggle.pull_kernel /
kernel_status / kernel_output / push_kernel — real Kaggle API calls, no
Chrome extension involved. It's also the concrete way to test the risk-tier
split you asked for: pull_kernel, kernel_status, kernel_output, and
push_kernel are all SAFE-tier (see autobot/agent/approval.py's
_SAFE_COMPUTER_CALL_PATTERNS) and should run through in every approval
mode without a pause — while a goal that reaches kaggle.submit(...) is
IRREVERSIBLE-tier and must always stop for a live "Allow", even in
trusted mode. This script deliberately never reaches submit() — there is
no --submit flag, on purpose; test that path by hand, once, watching for
the approval pause, then deny it so you don't burn a real submission
attempt just to prove the gate works.

Requires: ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY env vars)
already configured on this machine — this hits your real Kaggle account.

Usage:
  python demo_kaggle_run.py <kernel-slug>              # read-only: pull + check status + read output
  python demo_kaggle_run.py <kernel-slug> --push        # also push a trivial change, confirm a new run queues
  python demo_kaggle_run.py <kernel-slug> --goal "..."  # your own goal text, used verbatim

  kernel-slug looks like "your-username/your-kernel-name" — must be a
  kernel you own (pull_kernel/push_kernel act on your own account only).

Before running: make sure `autobot --server` (or `python -m autobot.main`)
is running. The Chrome extension does NOT need to be connected for this —
none of these actions go through the browser bridge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BACKEND = "http://127.0.0.1:8000"

READ_ONLY_GOAL_TEMPLATE = (
    "Pull the kernel '{kernel}' to ./kernel_test using computer.kaggle.pull_kernel. "
    "Then check its current run status with computer.kaggle.kernel_status. "
    "If the status shows a completed run, download its output with "
    "computer.kaggle.kernel_output to ./kernel_test_output and briefly describe "
    "what files came back. Do not push, submit, or modify anything — this is a "
    "read-only check. Call done with a one-paragraph summary of what you found."
)

PUSH_GOAL_TEMPLATE = (
    "Pull the kernel '{kernel}' to ./kernel_test using computer.kaggle.pull_kernel. "
    "Read the notebook/script source you just pulled. Make one small, harmless "
    "change to it (e.g. add a comment noting this was a test push, or a print "
    "statement) using your file tools, then push it back with "
    "computer.kaggle.push_kernel so a new run kicks off. Check "
    "computer.kaggle.kernel_status once to confirm it queued. Do not submit "
    "anything to a competition. Call done once you've confirmed the push queued "
    "a new run."
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "kernel", help="Kernel slug you own, e.g. 'your-username/your-kernel-name'"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Also push a trivial change and confirm a new run queues "
        "(still SAFE-tier, still not a competition submission)",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Override the goal entirely (used verbatim — the kernel slug "
        "argument is ignored when this is set)",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    if args.goal:
        goal = args.goal
    elif args.push:
        goal = PUSH_GOAL_TEMPLATE.format(kernel=args.kernel)
    else:
        goal = READ_ONLY_GOAL_TEMPLATE.format(kernel=args.kernel)

    try:
        _get("/api/health")
    except Exception as e:
        print(f"✗ Can't reach the Autobot backend at {BACKEND}: {e}")
        print("  Start it first:  autobot --server   (or: python -m autobot.main)")
        sys.exit(1)
    print("✓ Backend reachable. (Chrome extension connection not required for this run.)\n")

    print(f"Goal: {goal}\n")
    watched = "pull_kernel / kernel_status / kernel_output" + (
        " / push_kernel" if args.push else ""
    )
    print(
        f"Watch for: {watched} should all proceed with no approval pause, in "
        f"any mode — that's the SAFE-tier gate (approval.py's "
        f"_SAFE_COMPUTER_CALL_PATTERNS) actually working.\n"
    )

    status, resp = _post("/api/agent/run", {"goal": goal, "max_steps": args.max_steps})
    if status == 409:
        print(f"✗ {resp.get('detail', 'A run is already in progress.')}")
        print("  Wait for it to finish, or restart the backend, then try again.")
        sys.exit(1)
    if status != 200:
        print(f"✗ Could not start run (HTTP {status}): {resp}")
        sys.exit(1)

    run_id = resp.get("run_id")
    print(f"→ Run started: {run_id}\n{'-' * 70}")

    seen = 0
    # Kernel pull/push round-trips to Kaggle's API are slower than a DOM
    # read or a UIAutomation click — generous window vs. demo_agent_run.py's 180s.
    deadline = time.time() + 240
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

    print(f"{'-' * 70}\nFinal status: {last_status}")
    if last_status == "running":
        print("(Still running after 4 minutes — check the dashboard or /api/logs directly.)")


if __name__ == "__main__":
    main()
