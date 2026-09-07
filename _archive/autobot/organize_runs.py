"""
One-time (or occasional) organization of the runs folder.

- New runs already use human-readable folder names: plan_YYYY-MM-DD_HH-MM-SS
  with history.json, artifacts.json, screenshots/, console.log, about.txt

- This script finds old single-file runs (runs/*.json) and moves each into
  a folder with a readable name, so everything is "run folder + about.txt"
  and you can see at a glance what each run is.

Usage:
  python -m autobot.organize_runs       # dry run (print what would be done)
  python -m autobot.organize_runs --do  # actually move files and create about.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _safe_plan(plan_name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in plan_name).strip("._") or "run"


def _folder_name_from_json(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        plan = data.get("plan_name") or "run"
        started = data.get("started_at")
        if started:
            try:
                dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                return f"{_safe_plan(plan)}_{dt.strftime('%Y-%m-%d_%H-%M-%S')}"
            except Exception:
                pass
        return f"{_safe_plan(plan)}_migrated"
    except Exception:
        return None


def _folder_name_from_filename(path: Path) -> str:
    # Old format: 20260219_202921_640770_adapter_ui_call.json
    name = path.stem
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_\d+_(.+)", name)
    if m:
        y, mo, d, h, mi, s, plan = m.groups()
        return f"{_safe_plan(plan)}_{y}-{mo}-{d}_{h}-{mi}-{s}"
    return f"run_{name[:50]}"


def organize_runs(runs_dir: Path, dry_run: bool = True) -> int:
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        return 0

    count = 0
    for path in sorted(runs_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        folder_name = _folder_name_from_json(path) or _folder_name_from_filename(path)
        target_dir = runs_dir / folder_name
        if target_dir.exists():
            suffix = 1
            while (runs_dir / f"{folder_name}_{suffix}").exists():
                suffix += 1
            target_dir = runs_dir / f"{folder_name}_{suffix}"
        if dry_run:
            print(f"Would move: {path.name} -> {target_dir.name}/history.json")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / "history.json"
            path.rename(dest)
            try:
                data = json.loads(dest.read_text(encoding="utf-8"))
                started = data.get("started_at", "")
                finished = data.get("finished_at", "")
                success = data.get("success", False)
                completed = data.get("completed_steps", 0)
                total = data.get("total_steps", 0)
                plan = data.get("plan_name", "run")
                about_lines = [
                    f"Plan: {plan}",
                    f"Started: {started[:19].replace('T', ' ')} UTC" if started else "Started: (unknown)",
                    f"Finished: {finished[:19].replace('T', ' ')} UTC" if finished else "Finished: (unknown)",
                    f"Success: {success}",
                    f"Steps: {completed}/{total}",
                    "",
                    "Legacy run (migrated). Contents: history.json only (no screenshots/artifacts/console).",
                ]
                (target_dir / "about.txt").write_text("\n".join(about_lines), encoding="utf-8")
            except Exception:
                (target_dir / "about.txt").write_text(
                    f"Plan: (unknown)\nLegacy run (migrated). Contents: history.json only.",
                    encoding="utf-8",
                )
            print(f"Moved: {path.name} -> {target_dir.name}/")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize runs folder: put old JSON runs into readable-named folders.")
    parser.add_argument("--do", action="store_true", help="Actually perform moves (default is dry run).")
    parser.add_argument("--runs-dir", type=Path, default=Path.cwd() / "runs", help="Path to runs directory.")
    args = parser.parse_args()
    runs_dir = args.runs_dir
    if not runs_dir.is_absolute():
        runs_dir = Path.cwd() / runs_dir
    if args.do:
        n = organize_runs(runs_dir, dry_run=False)
        print(f"Migrated {n} run(s).")
    else:
        n = organize_runs(runs_dir, dry_run=True)
        print(f"Dry run: {n} run(s) would be migrated. Use --do to apply.")
    if n > 0 and not args.do:
        sys.exit(0)
    elif n == 0:
        sys.exit(0)


if __name__ == "__main__":
    main()
