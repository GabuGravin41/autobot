"""
Floating HUD Launcher — CLI & System Tray Command Palette for Autobot.
Allows instant goal entry and permission dial adjustments.
"""
from __future__ import annotations

import argparse
import os
import sys
import httpx

SERVER_URL = "http://localhost:8000"


def send_goal(goal: str, permission_level: str = "level_1_supervised"):
    """Submit a goal to the running Autobot server."""
    os.environ["AUTOBOT_PERMISSION_LEVEL"] = permission_level
    try:
        r = httpx.post(
            f"{SERVER_URL}/api/agent/run",
            json={"goal": goal, "max_steps": 25},
            timeout=10,
        )
        if r.status_code == 200:
            print(f"🚀 Goal submitted successfully to Autobot Server!")
            print(f"Goal: '{goal}'")
            print(f"Permission Level: {permission_level}")
        else:
            print(f"❌ Server error ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"❌ Connection error: Could not reach Autobot server at {SERVER_URL}. Is 'autobot --server' running?")
        print(f"Error details: {e}")


def main():
    parser = argparse.ArgumentParser(description="Autobot HUD & Command Palette Launcher")
    parser.add_argument("goal", nargs="?", help="Goal description for Autobot to execute")
    parser.add_argument(
        "--level",
        choices=["level_0_observer", "level_1_supervised", "level_2_full"],
        default="level_1_supervised",
        help="Permission Dial level",
    )
    args = parser.parse_args()

    if args.goal:
        send_goal(args.goal, args.level)
    else:
        print("🤖 Autobot Command HUD")
        print("-----------------------")
        print("Usage: python -m autobot.hud \"Your goal description here\" --level level_1_supervised")


if __name__ == "__main__":
    main()
