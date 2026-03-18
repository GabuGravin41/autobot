import logging
import os
import sys
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
            load_dotenv(_env_path, override=False)  # don't override vars already in environment
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


def _open_browser(url: str) -> None:
    """Open the dashboard in the user's default browser (best-effort)."""
    import threading, webbrowser, time
    def _open():
        time.sleep(1.5)  # give uvicorn a moment to bind
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    # Desktop app binds to localhost only (not exposed to network)
    host = os.getenv("AUTOBOT_HOST", "127.0.0.1")
    port = int(os.getenv("AUTOBOT_PORT", os.getenv("PORT", "8000")))
    url  = f"http://127.0.0.1:{port}"

    logger.info(f"Starting Autobot on {url}")

    # Auto-open dashboard in browser unless suppressed
    if os.getenv("AUTOBOT_NO_BROWSER", "").lower() not in ("1", "true", "yes"):
        _open_browser(url)

    try:
        uvicorn.run(
            "autobot.web.app:app",
            host=host,
            port=port,
            reload=False,
            log_level="warning",  # suppress uvicorn's own request logs (our app logs are enough)
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
