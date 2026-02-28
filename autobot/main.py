import os
import sys
from pathlib import Path

# Load .env from project root so AUTOBOT_* and XAI_API_KEY etc. are set before UI/engine start
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        print("Warning: python-dotenv not found. Environment variables from .env might not be loaded.")

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn not found. Please install it with 'pip install uvicorn'.")
    sys.exit(1)


def main() -> None:
    # Disable tkinter launch if any
    # from .ui import launch_ui
    # launch_ui()
    
    print("Starting Autobot Web Server on http://127.0.0.1:8000")
    try:
        uvicorn.run("autobot.web.app:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
