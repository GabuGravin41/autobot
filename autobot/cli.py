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
    """Load .env from project root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
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
    
    # 1. Install Playwright browsers
    print("📦 Installing Playwright browsers (chromium)...")
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        print("✅ Playwright ready.")
    except Exception as e:
        print(f"❌ Playwright setup failed: {e}")

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
