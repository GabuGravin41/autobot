"""
Run the tool_call_stress workflow from the command line.
Logs go to stdout and, after run_dir is known, to run_dir/console.log.
Use this to run the test when you're away and inspect runs/<stamp>_tool_call_stress/
for history.json, screenshots/, and artifacts.json when you return.

Usage:
  python -m autobot.run_stress
  python -m autobot.run_stress "+1234567890|https://docs.google.com/.../edit|C:/Users/.../Downloads/out.pdf|Hello"
  set AUTOBOT_STRESS_TOPIC=+1234|https://...|C:/path/to/pdf|Message
  python -m autobot.run_stress
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .engine import AutomationEngine
from .workflows import tool_call_stress_workflow


def _parse_topic(topic: str) -> tuple[str, str, str, str]:
    """Parse 'phone|docs_url|download_path|message' into 4 parts."""
    parts = topic.strip().split("|", 3)
    phone = (parts[0] or "").strip()
    docs_url = (parts[1] or "").strip()
    download_path = (parts[2] or "").strip()
    message = (parts[3] or "Autobot tool-calling test message").strip()
    return phone, docs_url, download_path, message


def run_stress(topic: str | None = None) -> dict:
    if topic is None:
        topic = os.getenv("AUTOBOT_STRESS_TOPIC", "")
    if not topic:
        topic = "|||Autobot tool-calling test message"
    phone, docs_url, download_path, message = _parse_topic(topic)

    log_lines: list[str] = []

    def logger(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    engine = AutomationEngine(logger=logger)
    plan = tool_call_stress_workflow(
        whatsapp_phone=phone,
        docs_existing_url=docs_url,
        download_check_path=download_path,
        outgoing_message=message,
    )
    try:
        result = engine.run_plan(plan)
        run_dir = result.state.get("run_dir", "")
        if run_dir:
            console_log_path = Path(run_dir) / "console.log"
            try:
                console_log_path.write_text("\n".join(log_lines), encoding="utf-8")
            except Exception:  # noqa: S110
                pass
            logger(f"Run folder: {run_dir}")
            logger(f"Console log: {console_log_path}")
        return {
            "success": result.success,
            "completed_steps": result.completed_steps,
            "total_steps": result.total_steps,
            "run_dir": result.state.get("run_dir"),
            "last_run_history_path": result.state.get("last_run_history_path"),
        }
    finally:
        engine.close()


def main() -> None:
    topic = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    try:
        out = run_stress(topic)
        if not out.get("success"):
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
