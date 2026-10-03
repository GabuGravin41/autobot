"""Resumable-run helpers: manifest tracking + atomic writes.

Design: every long stage writes `work_dir/manifest.json` like
{"task1": {"done": true, "rows": N, ...}, ...}. On restart with
resume=True, stages whose manifest entry is done AND whose output
file exists with size>0 are skipped. Per-panel checkpoints live in
work_dir/checkpoints/<stage>/<panel>.done + partial CSVs, so a kill
mid-panel only re-runs that panel, not everything.

All writes are atomic (tmp + os.replace) so Ctrl-C never leaves
half-written CSVs marked done.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def load_manifest(work_dir: Path) -> dict:
    p = work_dir / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_manifest(work_dir: Path, manifest: dict) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = work_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, work_dir / "manifest.json")


def mark_done(work_dir: Path, stage: str, info: dict | None = None) -> None:
    m = load_manifest(work_dir)
    m[stage] = {"done": True, "ts": time.time(), **(info or {})}
    save_manifest(work_dir, m)


def mark_started(work_dir: Path, stage: str) -> None:
    m = load_manifest(work_dir)
    m[stage] = {"done": False, "ts": time.time()}
    save_manifest(work_dir, m)


def is_done(work_dir: Path, stage: str, out_path: Path | None = None) -> bool:
    m = load_manifest(work_dir)
    e = m.get(stage)
    if not e or not e.get("done"):
        return False
    if out_path is not None:
        if not out_path.exists() or out_path.stat().st_size == 0:
            return False
    return True


def panel_done(work_dir: Path, stage: str, panel: str) -> bool:
    p = work_dir / "checkpoints" / stage / f"{panel}.done"
    return p.exists()


def mark_panel_done(work_dir: Path, stage: str, panel: str, rows: int = 0) -> None:
    d = work_dir / "checkpoints" / stage
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{panel}.done.tmp"
    tmp.write_text(json.dumps({"panel": panel, "rows": rows, "ts": time.time()}))
    os.replace(tmp, d / f"{panel}.done")


def atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)
