"""
Single source of truth for where Autobot keeps runtime state.

Before this module, three stores (ProjectRegistry, SkillDistiller,
EnvironmentMemory) each defaulted to `Path.cwd() / "autobot" / "knowledge"`,
so their data landed wherever the process happened to be started from.
Starting the daemon from a scheduled task, a different terminal folder, or
Antigravity's working directory silently created a second, empty copy of
every store. On top of that, the repo lives inside a OneDrive-synced folder,
which is a poor place for files an always-on process rewrites constantly
(sync conflicts, Files On-Demand placeholders).

Everything now lives under one home directory, outside the repo:

    AUTOBOT_HOME   (env var)   — if set, used as-is
    ~/.autobot                 — default (C:\\Users\\<you>\\.autobot on Windows)

Every store still accepts an explicit path argument, so tests pass a
tmp_path and never touch the real home directory.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_USER_EMAIL: str = "daltonomondi588@gmail.com"



def autobot_home() -> Path:
    """Root of all runtime state. Created on first use."""
    raw = os.getenv("AUTOBOT_HOME", "").strip()
    home = Path(raw).expanduser() if raw else Path.home() / ".autobot"
    home.mkdir(parents=True, exist_ok=True)
    return home


def state_dir(*parts: str) -> Path:
    """A subdirectory of autobot_home(), created on first use."""
    p = autobot_home().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def knowledge_dir(*parts: str) -> Path:
    return state_dir("knowledge", *parts)


def butler_db_path() -> Path:
    return autobot_home() / "butler.db"


def logs_dir() -> Path:
    return state_dir("logs")


def workspaces_dir() -> Path:
    """Scratch space for butler tasks that don't have a project folder of their own."""
    return state_dir("workspaces")


def repo_root() -> Path:
    """The source checkout this code is running from (NOT a place for state)."""
    return Path(__file__).resolve().parent.parent


def migrate_legacy_projects() -> list[str]:
    """
    One-time, non-destructive copy of project-registry files from the old
    repo-relative location into autobot_home(). Only copies files that don't
    already exist at the destination; never deletes the originals.

    Learned skills are deliberately NOT migrated: the old skills folder was
    written by a distiller that saved failed runs as "proven approaches"
    (see skill_distiller.py), so its contents can't be trusted wholesale.
    """
    copied: list[str] = []
    legacy = repo_root() / "autobot" / "knowledge" / "projects"
    if not legacy.is_dir():
        return copied
    dest = knowledge_dir("projects")
    for src in legacy.glob("*.json"):
        target = dest / src.name
        if not target.exists():
            shutil.copy2(src, target)
            copied.append(src.name)
    return copied
