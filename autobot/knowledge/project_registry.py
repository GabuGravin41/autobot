"""
Project Registry — persistent memory of the multiple AI-assisted projects
the user is juggling at once (VS Code + Claude Code, Antigravity x N), so
Autobot can check in on each one without the user re-explaining what it is
and what they want from it every single time.

This is the concrete piece of "understand exactly what I'm working on" from
the user's own framing of the orchestrator feature (Sep 2026): a project is
registered once with its working directory, which AI backend actually
drives it, and the user's own stated intent (kept verbatim, not
summarized/rewritten — see register()'s docstring for why that matters),
and from then on the orchestrator resumes that project's AI conversation by
session/conversation id rather than starting cold.

Storage pattern deliberately mirrors autobot/knowledge/skill_distiller.py:
one JSON file per tracked project under a knowledge/ subdirectory, a
dataclass with to_dict()/from_dict(), a filesystem-safe slug derived from
the project name. Kept consistent on purpose — this project's existing
convention for "small amount of state that needs to survive across runs
without a database" already works and is already tested elsewhere.
"""
from __future__ import annotations

import hashlib
import os
import threading
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The only two headless AI backends Autobot can actually drive today (see
# autobot/computer/claude_code_tool.py and antigravity_tool.py) — both CLI,
# neither GUI automation. Kept as a real validated set, not a free-text
# field, so a typo doesn't silently produce a TrackedProject the check-in
# loop can never dispatch (it would just KeyError deep inside the
# orchestrator later instead of failing clearly at registration time).
KNOWN_BACKENDS = ("claude_code", "antigravity")


@dataclass
class TrackedProject:
    """One project the user is working on with an AI backend's help.

    session_id means different things per backend — claude_code's
    resume_session_id vs antigravity's conversation_id — but both serve the
    identical purpose (resume this specific ongoing conversation instead of
    starting fresh), so one field covers both rather than two
    backend-specific ones the orchestrator would have to branch on.
    """

    name: str
    working_dir: str
    backend: str                       # one of KNOWN_BACKENDS
    intent: str                        # the user's own words — see register()
    created_at: str
    session_id: str | None = None
    last_checked_at: str | None = None
    last_status_summary: str | None = None
    intent_notes: list[str] = field(default_factory=list)  # later additions, oldest first

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "working_dir": self.working_dir,
            "backend": self.backend,
            "intent": self.intent,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "last_checked_at": self.last_checked_at,
            "last_status_summary": self.last_status_summary,
            "intent_notes": self.intent_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackedProject":
        return cls(**data)

    def full_intent_context(self) -> str:
        """The intent plus every note added since, joined for prompt
        injection — this is what the orchestrator hands to the AI backend
        as "here's what the user actually wants" on a check-in or
        follow-up prompt."""
        parts = [self.intent]
        parts.extend(self.intent_notes)
        return "\n".join(p for p in parts if p and p.strip())


class ProjectRegistry:
    """Register, look up, update, and list tracked projects."""

    def __init__(self, projects_dir: Path | None = None) -> None:
        if projects_dir is None:
            from autobot.paths import knowledge_dir, migrate_legacy_projects
            migrate_legacy_projects()
            projects_dir = knowledge_dir("projects")
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        name: str,
        working_dir: str,
        backend: str,
        intent: str,
    ) -> TrackedProject:
        """
        Register a new tracked project, or update an existing one's
        working_dir/backend/intent if the name is already registered.

        intent is stored VERBATIM — whatever the user actually typed or
        pasted describing the project, not a paraphrase or summary Autobot
        generated. This matters for the same reason it matters in
        conversation memory generally: a summary silently drops the parts
        the summarizer judged unimportant, and the user is the one who
        gets to decide what's important about their own project, not
        Autobot mid-registration. Re-registering an existing name REPLACES
        intent rather than appending — use add_intent_note() instead when
        the goal is "the user told me one more thing about this project",
        not "the user redefined what this project is."

        Raises ValueError for an unrecognized backend rather than silently
        storing it — see KNOWN_BACKENDS' docstring for why that fails loud
        here instead of quietly breaking the check-in loop later.
        """
        if backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r} — must be one of {KNOWN_BACKENDS}. "
                f"Autobot can only drive a project through a real headless CLI; "
                f"there is deliberately no GUI-automation backend option."
            )
        if not name or not name.strip():
            raise ValueError("register: name is required")
        if not intent or not intent.strip():
            raise ValueError(
                "register: intent is required — an unregistered-intent project "
                "defeats the entire point of this registry, which is knowing "
                "what the user actually wants from it."
            )

        existing = self.get(name)
        project = TrackedProject(
            name=name.strip(),
            working_dir=working_dir,
            backend=backend,
            intent=intent,
            created_at=existing.created_at if existing else datetime.now(timezone.utc).isoformat(),
            session_id=existing.session_id if existing else None,
            last_checked_at=existing.last_checked_at if existing else None,
            last_status_summary=existing.last_status_summary if existing else None,
            intent_notes=existing.intent_notes if existing else [],
        )
        self._save(project)
        return project

    def get(self, name: str) -> TrackedProject | None:
        path = self._path_for(name)
        if not path.exists():
            return None
        try:
            return TrackedProject.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"Could not load tracked project {path}: {e}")
            return None

    def list_all(self) -> list[TrackedProject]:
        """All tracked projects, most-recently-checked-in first (never
        checked in yet sorts last) — this is the natural iteration order
        for an orchestrator deciding which project to check on next."""
        projects: list[TrackedProject] = []
        for filepath in self.projects_dir.glob("*.json"):
            try:
                projects.append(TrackedProject.from_dict(json.loads(filepath.read_text(encoding="utf-8"))))
            except Exception as e:
                logger.warning(f"Could not load tracked project {filepath}: {e}")
        projects.sort(key=lambda p: p.last_checked_at or "", reverse=True)
        return projects

    def add_intent_note(self, name: str, note: str) -> TrackedProject:
        """Append one more thing the user said about this project, without
        touching the original intent text. Raises KeyError if the project
        isn't registered — silently no-op-ing on a typo'd name would hide
        exactly the kind of 'said something and it went nowhere' failure
        this whole feature exists to prevent."""
        project = self._require(name)
        if note and note.strip():
            project.intent_notes.append(note.strip())
            self._save(project)
        return project

    def update_session(self, name: str, session_id: str) -> TrackedProject:
        """Remember the AI backend's session/conversation id after a run,
        so the next check-in resumes it instead of starting cold."""
        project = self._require(name)
        project.session_id = session_id
        self._save(project)
        return project

    def record_check_in(self, name: str, status_summary: str) -> TrackedProject:
        """Record the result of a check-in (see the orchestrator loop) —
        updates last_checked_at to now and stores the summary text."""
        project = self._require(name)
        project.last_checked_at = datetime.now(timezone.utc).isoformat()
        project.last_status_summary = status_summary
        self._save(project)
        return project

    def forget(self, name: str) -> bool:
        """Remove a project from tracking. Returns False if it wasn't
        tracked (not an error — matches dict.pop-with-default semantics
        rather than raising for what's usually a harmless double-call)."""
        path = self._path_for(name)
        if not path.exists():
            return False
        path.unlink()
        return True

    # ── internals ────────────────────────────────────────────────────────

    def _require(self, name: str) -> TrackedProject:
        project = self.get(name)
        if project is None:
            raise KeyError(f"No tracked project named {name!r}. Registered: {[p.name for p in self.list_all()]}")
        return project

    def _save(self, project: TrackedProject) -> None:
        # Atomic write (temp file + rename), matching kaggle_watchdog.py's
        # KaggleJobLedger.save(). Previously a plain write_text() — flagged
        # in Round 6 as "not triggered by today's only caller's genuinely
        # sequential for-loop, but worth a real decision before this
        # registry gets a concurrent caller." Round 8's orchestrator_dispatch.py
        # is exactly that concurrent caller (multiple asyncio tasks calling
        # record_check_in/update_session at once via dispatch_on_all) — each
        # task writes a DIFFERENT project's file, so there's no cross-task
        # contention on one file, but a crash or interrupt mid-write to any
        # single file must never leave that one project's JSON half-written
        # and unreadable on the next load(). Rename is atomic on both POSIX
        # and Windows.
        path = self._path_for(project.name)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    def _path_for(self, name: str) -> Path:
        return self.projects_dir / f"{self._safe_name(name)}.json"

    @staticmethod
    def _safe_name(name: str) -> str:
        """Filesystem-safe slug — same approach as skill_distiller.py's
        _safe_name(), including Windows' illegal-character set, since this
        project runs on Windows and a project name typed with any of
        : * ? " < > | would otherwise OSError on save.

        A short hash of the exact (case-folded) name is appended on
        purpose: the human-readable part alone collapses whitespace and
        punctuation to a single "_", so two DIFFERENT project names —
        e.g. "Foo Bar" and "Foo-Bar" — reduce to the identical slug
        "foo_bar". Without the hash, registering the second name resolves
        to the SAME underlying JSON file as the first (_path_for() is a
        pure function of this return value), so register() would silently
        overwrite an unrelated project's stored session_id/intent/history
        instead of creating a second, separate one — see
        tests/test_project_registry.py::TestFilesystemSafety::
        test_names_that_collide_after_slugging_do_not_overwrite_each_other
        for the regression this guards against. Hashing the case-folded
        name (not the original) preserves the pre-existing behavior that
        re-registering under a different case of the SAME name (e.g.
        "Seqoy" vs "SEQOY") still resolves to the same project."""
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        slug = (slug or "project")[:60]
        digest = hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:8]
        return f"{slug}_{digest}"
