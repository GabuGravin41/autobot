import sys

from .main import main


def _entry() -> None:
    # `python -m autobot butler ...` / `python -m autobot kaggle ...` go to the
    # CLI; everything else keeps starting the backend server as before.
    if len(sys.argv) > 1 and sys.argv[1] in ("butler", "kaggle"):
        from .cli import main as cli_main
        cli_main()
    else:
        main()


if __name__ == "__main__":
    _entry()
