import os
import sys
from pathlib import Path

# Support PyInstaller bundled app: sys._MEIPASS points to the temp extraction dir
_bundle_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))

# Load .env: check CWD first (user's custom .env), then bundle default
_env_candidates = [
    Path.cwd() / ".env",                       # User's working directory
    Path(__file__).resolve().parent.parent / ".env",  # Development layout
    _bundle_dir / ".env_default" / ".env",      # PyInstaller bundle
]
for _env_path in _env_candidates:
    if _env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path)
            break
        except ImportError:
            print("Warning: python-dotenv not found. Environment variables from .env might not be loaded.")
            break

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn not found. Please install it with 'pip install uvicorn'.")
    sys.exit(1)


def main() -> None:
    # Disable tkinter launch if any
    # from .ui import launch_ui
    # launch_ui()
    
    host = os.getenv("AUTOBOT_HOST", "0.0.0.0")
    port = int(os.getenv("AUTOBOT_PORT", "8000"))
    print(f"Starting Autobot Web Server on http://{host}:{port}")
    print(f"  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://0.0.0.0:{port} (accessible from other devices on same network)")
    try:
        uvicorn.run("autobot.web.app:app", host=host, port=port, reload=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
