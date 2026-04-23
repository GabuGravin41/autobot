import argparse
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

# Support PyInstaller bundled app: sys._MEIPASS points to the temp extraction dir
_bundle_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))

# Load .env: check standard locations in priority order
_env_candidates = [
    Path(os.getenv("AUTOBOT_ENV_PATH", "")) if os.getenv("AUTOBOT_ENV_PATH") else None,
    Path.home() / ".autobot" / ".env",         # persistent user config (AppImage)
    Path.cwd() / ".env",                        # development / custom location
    Path(__file__).resolve().parent.parent / ".env",  # project root
    _bundle_dir / ".env.example",               # bundled template (read-only fallback)
]
for _env_path in (p for p in _env_candidates if p):
    if _env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path, override=False)
        except ImportError:
            pass
        break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autobot")

try:
    import uvicorn
except ImportError:
    logger.error("uvicorn not found. Run: pip install uvicorn")
    sys.exit(1)

# Resolve paths
_project_root = Path(__file__).resolve().parent.parent
_frontend_dir  = _project_root / "frontend"
_dist_dir      = _frontend_dir / "dist"


def _npm(*args: str, stream: bool = False) -> subprocess.Popen | None:
    """Run an npm command in the frontend directory."""
    cmd = ["npm", *args]
    if not stream:
        result = subprocess.run(cmd, cwd=str(_frontend_dir), capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"npm {' '.join(args)} failed:\n{result.stderr.strip()}")
            return None
        return result
    return subprocess.Popen(
        cmd,
        cwd=str(_frontend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _build_frontend(force: bool = False) -> bool:
    """Build the frontend bundle. Returns True on success."""
    if not _frontend_dir.exists():
        logger.warning("frontend/ directory not found — skipping build.")
        return False

    if not force and _dist_dir.exists() and any(_dist_dir.iterdir()):
        logger.info("Frontend dist/ already built — skipping (use --rebuild to force).")
        return True

    logger.info("Building frontend bundle (this takes ~20s the first time)...")
    result = _npm("run", "build:fast")
    if result is None:
        logger.error("Frontend build failed — serving without UI.")
        return False

    logger.info("Frontend built successfully.")
    return True


def _start_watch(stop_event: threading.Event) -> None:
    """Run 'vite build --watch' in the background; killed when stop_event is set."""
    logger.info("Starting frontend watcher — UI rebuilds automatically when you edit files.")
    proc = _npm("run", "watch", stream=True)
    if proc is None:
        return
    stop_event.wait()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autobot — desktop automation agent")
    parser.add_argument("--rebuild",  action="store_true", help="Force rebuild of frontend before starting")
    parser.add_argument("--no-watch", action="store_true", help="Don't run the frontend file watcher")
    args = parser.parse_args()

    # AUTOBOT_HOST > HOST > default 0.0.0.0 (expose to LAN so phones on same WiFi can connect)
    host = os.getenv("AUTOBOT_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("AUTOBOT_PORT", os.getenv("PORT", "8000")))
    local_url = f"http://127.0.0.1:{port}"

    # Resolve the LAN IP so we can print the phone address
    try:
        import socket
        _lan_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        _lan_ip = host

    # ── 1. Build frontend (if needed) ─────────────────────────────────────────
    _build_frontend(force=args.rebuild)

    # ── 2. Start frontend watcher in background thread ─────────────────────────
    _stop = threading.Event()
    no_watch = args.no_watch or os.getenv("AUTOBOT_NO_WATCH", "").lower() in ("1", "true", "yes")
    if not no_watch and _frontend_dir.exists():
        watcher_thread = threading.Thread(target=_start_watch, args=(_stop,), daemon=True)
        watcher_thread.start()

    # ── 3. Print addresses (no auto-open — navigate there yourself) ────────────
    logger.info(f"Local:  {local_url}")
    logger.info(f"Phone:  http://{_lan_ip}:{port}  (same WiFi)")

    # ── 4. Launch backend (blocks until Ctrl+C) ────────────────────────────────
    try:
        uvicorn.run(
            "autobot.web.app:app",
            host=host,
            port=port,
            reload=False,
            log_level="warning",
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
