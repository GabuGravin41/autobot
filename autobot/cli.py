"""
Autobot CLI — run tasks from the command line.

Usage:
    autobot "search for the latest AI papers"
    autobot --server                  # Start the dashboard server
    autobot --version                 # Show version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_env() -> None:
    """Load .env from project root.

    encoding="utf-8-sig" is deliberate and load-bearing on Windows. Windows
    PowerShell 5.1's `Out-File -Encoding utf8` writes a UTF-8 BOM, and with a
    plain utf-8 read that BOM becomes part of the FIRST variable's name —
    "﻿OPENROUTER_API_KEY" instead of "OPENROUTER_API_KEY". The result is
    maddening to debug: every setting in the file works except the first one,
    which silently reads as unset. "utf-8-sig" strips the BOM if present and
    is a no-op otherwise.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, encoding="utf-8-sig")
        except ImportError:
            pass


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        prog="autobot",
        description="Autobot — A sovereign digital agent with full computer control.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Natural language task to execute (e.g. 'open kaggle and read competition titles')",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the Autobot dashboard web server",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run initial setup (install playwright browsers, etc.)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Diagnose the environment (deps, Chrome/CDP, API keys) without running a task",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the web server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        print("autobot 0.1.0")
        return

    if args.doctor:
        from autobot.diagnostics import main as doctor_main
        sys.exit(doctor_main())

    if args.setup:
        _run_setup()
        return

    if args.server or args.task is None:
        _start_server(args.host, args.port)
    else:
        _run_task(args.task)


def _run_setup() -> None:
    """Run initial environment setup."""
    print("🛠️ Running Autobot Setup...")

    # NOTE: We deliberately do NOT run `playwright install chromium`.
    # Autobot drives the user's REAL Chrome (with their real logins) by
    # launching it with --remote-debugging-port and attaching via
    # connect_over_cdp(). It never calls chromium.launch(), so Playwright's
    # ~150MB bundled browser is never used. Downloading it is pure waste, and
    # it fails outright on networks that intercept TLS (corporate proxies,
    # HTTPS-scanning antivirus) because Playwright's bundled Node has its own
    # CA store that doesn't include the intercepting certificate.
    print("ℹ️  Skipping Playwright browser download — Autobot uses your real Chrome.")

    # Verify the Playwright *driver* imports (this is what we actually need).
    try:
        import playwright  # noqa: F401
        print("✅ Playwright driver available.")
    except ImportError:
        print("❌ Playwright not installed. Run: pip install -r requirements.txt")

    # 2. Check for frontend build
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if not frontend_dist.exists():
        print("💡 Tip: The dashboard frontend is not built. Run 'npm run build' in the /frontend folder to enable the web UI.")
    else:
        print("✅ Frontend build detected.")

    print("\n🚀 Setup complete. Start the dashboard with: autobot --server")


def _start_server(host: str, port: int) -> None:
    """Start the FastAPI dashboard server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not found. Install with: pip install autobot[all]")
        sys.exit(1)

    # Check build status to warn user
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if not frontend_dist.exists():
        print("⚠️  Warning: Frontend 'dist' folder not found. The dashboard will show a fallback page.")
        print("   To fix, run 'npm install && npm run build' in the /frontend directory.")

    print(f"Starting Autobot Dashboard on http://{host}:{port}")
    # Disable reload in 'packaged' mode for stability, but we can keep it for now
    uvicorn.run("autobot.web.app:app", host=host, port=port, reload=False)


def _run_task(task: str) -> None:
    """Run a single task from the command line via AgentRunner."""
    import asyncio

    async def _execute() -> None:
        from autobot.agent.runner import AgentRunner

        print(f"Autobot executing: {task}")
        print("─" * 50)

        runner = AgentRunner.from_env(
            log_callback=lambda msg: print(f"  {msg}"),
        )
        try:
            result = await runner.run(task)
            print("─" * 50)
            print(f"Done: {result}" if result else "Task finished.")
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as e:
            print(f"Failed: {e}")
            sys.exit(1)

    asyncio.run(_execute())


if __name__ == "__main__":
    main()
