import os
from pathlib import Path

# Load .env from project root so AUTOBOT_* and XAI_API_KEY etc. are set before UI/engine start
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

from .ui import launch_ui


def main() -> None:
    launch_ui()


if __name__ == "__main__":
    main()
