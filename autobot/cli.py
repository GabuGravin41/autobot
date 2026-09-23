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


def _make_stdout_unicode_safe() -> None:
    """Stop the CLI from dying on its own log characters.

    Autobot's output is full of box-drawing rules and emoji status markers.
    Windows Terminal negotiates UTF-8 so they render fine there, but a plain
    console — or ANY redirect to a file or pipe — falls back to cp1252, where
    printing them raises UnicodeEncodeError and takes the whole run down
    before a single agent step executes:

        print("\\u2500" * 50)
        UnicodeEncodeError: 'charmap' codec can't encode characters...

    errors="replace" matters as much as the encoding: if a console genuinely
    cannot render a glyph we want a '?' in the log, never a crashed run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # not a real TTY / already wrapped — nothing to do


def _use_system_certificates() -> None:
    """Make Python trust the OS certificate store, like browsers do.

    On networks that intercept TLS (campus/corporate proxies, antivirus with
    HTTPS scanning), the interceptor installs its root CA into the Windows
    certificate store. Chrome therefore works fine, while Python fails with
    CERTIFICATE_VERIFY_FAILED — because Python ships its own bundled CA list
    (certifi) and never consults the OS store.

    `truststore` redirects Python's TLS verification to the OS store, so the
    same certificates the rest of the machine already trusts are honoured
    here too. This keeps verification ON — unlike disabling it, which would
    hand the API key to whatever is doing the intercepting.

    Optional dependency: if it isn't installed we carry on unchanged.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
        logger_msg = "Using the OS certificate store for TLS verification."
    except ImportError:
        return
    except Exception as e:
        logger_msg = f"Could not enable OS certificate store ({e}); using bundled CAs."
    import logging
    logging.getLogger(__name__).debug(logger_msg)


def main() -> None:
    _make_stdout_unicode_safe()
    _use_system_certificates()
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
        "--jobs",
        action="store_true",
        help="Show the Kaggle job ledger, poll for status changes, and exit "
             "(a one-shot check for when 'autobot --server' isn't running to "
             "watch jobs in the background)",
    )
    parser.add_argument(
        "--register-project",
        metavar="NAME",
        help="Register or update a tracked project (requires --project-dir, "
             "--project-backend, --project-intent)",
    )
    parser.add_argument("--project-dir", help="Working directory, for --register-project")
    parser.add_argument(
        "--project-backend", choices=["claude_code", "antigravity"],
        help="AI backend driving the project, for --register-project",
    )
    parser.add_argument(
        "--project-intent", help="What you want from this project, in your own words "
        "(stored verbatim) — for --register-project",
    )
    parser.add_argument(
        "--list-projects", action="store_true",
        help="List every tracked project and its last known status",
    )
    parser.add_argument(
        "--check-in", metavar="NAME", nargs="?", const="__ALL__",
        help="Ask one tracked project (or all, if no name given) for a read-only "
             "status update",
    )
    parser.add_argument(
        "--dispatch", nargs=2, metavar=("NAME", "INSTRUCTION"),
        help="Send a real instruction to one tracked project's AI backend "
             "(read-only by default — see README.md for how to allow writes)",
    )
    parser.add_argument(
        "--dispatch-all", metavar="FILE",
        help="Send instructions to multiple tracked projects CONCURRENTLY, from a "
             "JSON file mapping project name -> instruction (optionally "
             "{\"instructions\": {...}, \"permission_modes\": {...}, "
             "\"skip_permissions\": {...}} for per-project write access)",
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

    if args.jobs:
        _show_jobs()
        return

    if args.register_project:
        _register_project(args.register_project, args.project_dir, args.project_backend, args.project_intent)
        return

    if args.list_projects:
        _list_projects()
        return

    if args.check_in:
        _check_in(None if args.check_in == "__ALL__" else args.check_in)
        return

    if args.dispatch:
        _dispatch(args.dispatch[0], args.dispatch[1])
        return

    if args.dispatch_all:
        _dispatch_all(args.dispatch_all)
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

    # NOTE: We deliberately do NOT run `playwright install chromium`.
    # As of the Round 5 CDP retirement (see ROADMAP.md), Autobot no longer
    # attaches to Chrome via --remote-debugging-port / connect_over_cdp() at
    # all — browser perception now goes through the Autobot Chrome extension
    # polling an already-open, already-logged-in Chrome instead. Playwright's
    # ~150MB bundled browser was never used even in the CDP era (it only ever
    # attached to the user's real Chrome, never launched its own), so
    # downloading it remains pure waste, and it still fails outright on
    # networks that intercept TLS (corporate proxies, HTTPS-scanning
    # antivirus) because Playwright's bundled Node has its own CA store that
    # doesn't include the intercepting certificate.
    print("Skipping Playwright browser download - Autobot uses your real Chrome via its extension.")

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


def _show_jobs() -> None:
    """
    One-shot Kaggle job ledger dump + poll — `autobot --server`'s
    background watchdog (see web/app.py's _kaggle_watchdog_loop) does this
    continuously while the dashboard is running, but a lot of real usage
    (a bare `autobot "<task>"` CLI run, or Antigravity/Claude Code driving
    Kaggle directly through kaggle_lab/-style scripts) never starts the
    server at all. This gives the same visibility without it: read the
    ledger, actively check every non-terminal job's current status once,
    and print what's there — same on-disk file, same source of truth,
    just polled synchronously instead of on a background schedule.
    """
    from autobot.computer.kaggle_watchdog import DEFAULT_LEDGER_PATH, KaggleJobLedger, poll_pending

    ledger = KaggleJobLedger(DEFAULT_LEDGER_PATH)
    jobs = ledger.all()
    if not jobs:
        print(f"No Kaggle jobs recorded yet in {DEFAULT_LEDGER_PATH}")
        print("(jobs are registered automatically by kaggle_tool.py's push_kernel())")
        return

    pending = ledger.pending()
    if pending:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            changed = poll_pending(api, ledger)
            if changed:
                print(f"Status changed for {len(changed)} job(s):")
                for j in changed:
                    print(f"  {j.kernel}: -> {j.status}")
                print()
        except Exception as e:
            print(f"(could not poll live status: {e} — showing last known state)\n")

    for job in ledger.all():
        marker = "OK" if job.status == "complete" else ("FAIL" if job.status == "error" else "...")
        print(f"[{marker}] {job.kernel}  status={job.status}  liveness_verified={job.liveness_verified}"
              + (f"  competition={job.competition}" if job.competition else ""))
        if job.status == "error" and job.error_log:
            print(f"       error: {job.error_log[:200]}")


def _register_project(name: str, working_dir: str | None, backend: str | None, intent: str | None) -> None:
    """
    Register or update a tracked project — the missing first step that
    made the whole multi-project orchestrator (project_registry.py +
    orchestrator_checkin.py, built Round 6) unreachable from any real run
    until Round 8: the registry and check-in logic were fully built and
    tested, but nothing in cli.py or web/app.py ever called
    ProjectRegistry.register() at all, so there was structurally no way to
    get a project INTO the registry outside of a test file. This command
    fixes that specific gap.
    """
    if not working_dir or not backend or not intent:
        print("--register-project requires --project-dir, --project-backend, and --project-intent")
        sys.exit(1)

    from autobot.knowledge.project_registry import ProjectRegistry

    try:
        project = ProjectRegistry().register(name, working_dir, backend, intent)
    except ValueError as e:
        print(f"Could not register project: {e}")
        sys.exit(1)

    print(f"Registered {project.name!r} ({project.backend}) at {project.working_dir}")
    print(f"Intent: {project.intent}")


def _list_projects() -> None:
    from autobot.knowledge.project_registry import ProjectRegistry

    projects = ProjectRegistry().list_all()
    if not projects:
        print("No tracked projects yet. Register one with:")
        print('  autobot --register-project NAME --project-dir PATH '
              '--project-backend claude_code --project-intent "what you want from it"')
        return

    for p in projects:
        checked = p.last_checked_at or "never checked in"
        print(f"[{p.backend}] {p.name}  ({p.working_dir})")
        print(f"    intent: {p.intent}")
        print(f"    last check-in: {checked}")
        if p.last_status_summary:
            print(f"    last status: {p.last_status_summary[:200]}")


def _check_in(name: str | None) -> None:
    """Read-only status check, one project or all — see
    orchestrator_checkin.py's module docstring for why this stays
    deliberately separate from --dispatch (a real instruction)."""
    import asyncio
    from autobot.agent.orchestrator_checkin import OrchestratorCheckIn

    orchestrator = OrchestratorCheckIn()

    async def _run() -> None:
        if name:
            results = [await orchestrator.check_in_on(name)]
        else:
            results = await orchestrator.check_in_on_all()
        if not results:
            print("No tracked projects to check in on. See --list-projects.")
            return
        for r in results:
            marker = "OK" if r.ok else "FAIL"
            print(f"[{marker}] {r.project_name} ({r.backend})")
            print(f"    {r.summary if r.ok else r.error}")

    asyncio.run(_run())


def _dispatch(name: str, instruction: str) -> None:
    """Send ONE real instruction to ONE tracked project. Read-only by
    default (permission_mode='plan' / skip_permissions=False, inherited
    from the bridge functions' own safe defaults — see
    orchestrator_dispatch.py's module docstring) — this CLI command
    doesn't currently expose a way to request write access; do that
    programmatically via OrchestratorDispatch.dispatch_on(...) if a
    project genuinely needs it, so that decision is made deliberately in
    code review, not as a casually-typed CLI flag."""
    import asyncio
    from autobot.agent.orchestrator_dispatch import OrchestratorDispatch

    async def _run() -> None:
        result = await OrchestratorDispatch().dispatch_on(name, instruction)
        marker = "OK" if result.ok else "FAIL"
        print(f"[{marker}] {result.project_name} ({result.backend})")
        print(result.summary if result.ok else result.error)

    asyncio.run(_run())


def _dispatch_all(instructions_file: str) -> None:
    """
    Run multiple tracked projects CONCURRENTLY — the "run multiple
    projects like OpenClaw" capability (Round 8). Reads a JSON file, one of:

        {"project-a": "do X", "project-b": "do Y"}

    or, for per-project write-access overrides:

        {
          "instructions": {"project-a": "do X", "project-b": "fix the bug"},
          "permission_modes": {"project-b": "acceptEdits"},
          "skip_permissions": {}
        }

    Bounded by AUTOBOT_ORCHESTRATOR_MAX_CONCURRENT (default 3) — see
    orchestrator_dispatch.py's module docstring for why that's a separate,
    softer limit from kaggle_watchdog.py's GPU/CPU slot enforcement.
    """
    import asyncio
    import json as _json
    from autobot.agent.orchestrator_dispatch import OrchestratorDispatch

    try:
        raw = _json.loads(Path(instructions_file).read_text(encoding="utf-8-sig"))
    except (OSError, _json.JSONDecodeError) as e:
        print(f"Could not read {instructions_file}: {e}")
        sys.exit(1)

    if "instructions" in raw:
        instructions = raw["instructions"]
        permission_modes = raw.get("permission_modes", {})
        skip_permissions = raw.get("skip_permissions", {})
    else:
        instructions = raw
        permission_modes = {}
        skip_permissions = {}

    if not instructions:
        print(f"{instructions_file} lists no projects to dispatch to.")
        return

    async def _run() -> None:
        results = await OrchestratorDispatch().dispatch_on_all(
            instructions, permission_modes=permission_modes, skip_permissions=skip_permissions,
        )
        for r in results:
            marker = "OK" if r.ok else "FAIL"
            print(f"[{marker}] {r.project_name} ({r.backend})")
            print(f"    {r.summary if r.ok else r.error}")

    asyncio.run(_run())


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
