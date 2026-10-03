"""
Small Kaggle command line for workers (Claude Code / Antigravity) to use
inside competition folders, so every kernel goes through Autobot's ledger,
slot limits and liveness check instead of a bare `kaggle kernels push`.

    python <repo>/autobot/kaggle_cli.py push <kernel_dir> [--competition SLUG] [--no-liveness]
    python <repo>/autobot/kaggle_cli.py output <owner/slug> -p <dir>
    python <repo>/autobot/kaggle_cli.py status <owner/slug>
    python <repo>/autobot/kaggle_cli.py jobs

Also reachable as `autobot kaggle ...` / `python -m autobot kaggle ...`.
Runnable as a plain script from any folder: it puts the repo on sys.path.
There is deliberately no `submit` command here — submissions go through
the butler's approval inbox.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def script_path() -> str:
    return str(Path(__file__).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autobot kaggle", description="Kaggle helpers that keep Autobot's job ledger accurate.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="Push a kernel folder (enforces GPU/CPU slot limits, records the job)")
    p.add_argument("kernel_dir")
    p.add_argument("--competition")
    p.add_argument("--no-liveness", action="store_true", help="Skip the 30-60s start-up check")
    o = sub.add_parser("output", help="Download a kernel's output files and log")
    o.add_argument("kernel")
    o.add_argument("-p", "--path", default="./kernel_output")
    s = sub.add_parser("status", help="Live status of one kernel")
    s.add_argument("kernel")
    sub.add_parser("jobs", help="Every kernel Autobot has pushed, with live status where possible")
    args = parser.parse_args(argv)

    try:
        from autobot.computer.kaggle_tool import Kaggle
        k = Kaggle()
        if args.cmd == "push":
            print(k.push_kernel(args.kernel_dir, verify_liveness=not args.no_liveness, competition=args.competition))
        elif args.cmd == "output":
            print(k.kernel_output(args.kernel, args.path))
        elif args.cmd == "status":
            from autobot.computer.kaggle_watchdog import _normalize_status, failure_message
            raw = k._get_api().kernels_status(args.kernel)
            msg = failure_message(raw)
            print(f"{args.kernel}: {_normalize_status(raw)}" + (f" — {msg}" if msg else ""))
        elif args.cmd == "jobs":
            from autobot.cli import _show_jobs
            _show_jobs()
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
