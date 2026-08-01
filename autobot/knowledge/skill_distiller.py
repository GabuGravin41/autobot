"""
Skill Distiller — Distills successful execution trajectories into reusable, learned skills.

Prevents Autobot from doing blind searching on tasks it has already solved before.
When a task completes successfully, SkillDistiller extracts proven action sequences,
lessons learned, and edge-case fallbacks, saving them into persistent skill files.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LearnedSkill:
    """A learned, reusable skill distilled from a successful run."""

    name: str
    description: str
    keywords: list[str]
    prerequisites: list[str]
    proven_steps: list[dict[str, Any]]
    lessons_learned: list[str]
    created_at: str
    success_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "prerequisites": self.prerequisites,
            "proven_steps": self.proven_steps,
            "lessons_learned": self.lessons_learned,
            "created_at": self.created_at,
            "success_count": self.success_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedSkill:
        return cls(**data)


class SkillDistiller:
    """
    Distills, stores, retrieves, and injects learned skills into Autobot runs.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir = skills_dir or (Path.cwd() / "autobot" / "knowledge" / "skills")
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def save_skill(self, skill: LearnedSkill) -> Path:
        """Save a distilled learned skill to JSON file."""
        filepath = self.skills_dir / f"{self._safe_name(skill.name)}.json"
        filepath.write_text(json.dumps(skill.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"🎓 Learned Skill Saved: '{skill.name}' at '{filepath}'")
        return filepath

    @staticmethod
    def _safe_name(name: str) -> str:
        """Filesystem-safe slug for a skill name.

        Skill names come from goal text, which routinely contains characters
        that are illegal in Windows filenames (: * ? " < > |) — the previous
        version only handled spaces and slashes, so a goal like
        'Research: perovskites?' produced an OSError on save.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return (slug or "skill")[:80]

    def distill_from_run(
        self,
        goal: str,
        history: list[Any],
        result: str,
    ) -> LearnedSkill | None:
        """
        Turn a successful run's step history into a reusable skill.

        Only the steps whose actions actually SUCCEEDED are recorded, so a
        replayed skill teaches the working path rather than replaying the
        agent's dead ends. Failed actions become 'lessons learned' instead —
        they're the more valuable half, since they tell the next run what not
        to waste steps on.

        Returns the saved skill, or None if there was nothing worth saving.
        If a skill for this goal already exists, its success_count is bumped
        and the proven steps are replaced only when the new run was shorter.
        """
        proven_steps: list[dict[str, Any]] = []
        lessons: list[str] = []

        for entry in history:
            successful = [r for r in entry.action_results if r.success]
            failed = [r for r in entry.action_results if not r.success]

            for r in failed:
                if r.error:
                    lessons.append(f"'{r.action_name}' failed: {r.error[:160]}")

            if successful:
                proven_steps.append({
                    "goal": entry.agent_output.next_goal,
                    "actions": [r.action_name for r in successful],
                    "url": entry.url_after,
                })

        if not proven_steps:
            logger.debug("Nothing to distill — no successful steps in this run.")
            return None

        name = goal.strip()[:60]
        existing = self._load_by_name(name)
        if existing:
            existing.success_count += 1
            # Prefer the shorter proven path — that's the whole point of learning.
            if len(proven_steps) < len(existing.proven_steps):
                existing.proven_steps = proven_steps
            existing.lessons_learned = self._dedupe(existing.lessons_learned + lessons)[:10]
            self.save_skill(existing)
            return existing

        skill = LearnedSkill(
            name=name,
            description=f"Proven approach for: {goal[:200]}",
            keywords=self._keywords(goal),
            prerequisites=[],
            proven_steps=proven_steps,
            lessons_learned=self._dedupe(lessons)[:10],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.save_skill(skill)
        return skill

    def _load_by_name(self, name: str) -> LearnedSkill | None:
        path = self.skills_dir / f"{self._safe_name(name)}.json"
        if not path.exists():
            return None
        try:
            return LearnedSkill.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.debug(f"Could not load existing skill {path}: {e}")
            return None

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for i in items:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    @staticmethod
    def _keywords(goal: str) -> list[str]:
        """Content words from the goal, used by find_matching_skill().

        Stopwords are stripped because find_matching_skill() returns on the
        FIRST keyword hit — leaving 'the'/'and'/'to' in would make unrelated
        skills match nearly every goal.
        """
        stop = {
            "the", "and", "for", "with", "from", "into", "that", "this", "then",
            "open", "http", "https", "www", "com", "use", "using", "get", "make",
            "a", "an", "of", "to", "in", "on", "at", "by", "it", "is", "are",
        }
        words = re.findall(r"[a-z0-9]{3,}", goal.lower())
        return SkillDistiller._dedupe([w for w in words if w not in stop])[:8]

    def find_matching_skill(self, goal: str) -> LearnedSkill | None:
        """
        Find a previously learned skill that matches the current goal.
        """
        goal_lower = goal.lower()
        for filepath in self.skills_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                skill = LearnedSkill.from_dict(data)

                # Check keyword match
                for kw in skill.keywords:
                    if kw.lower() in goal_lower:
                        logger.info(f"💡 Learned Skill Match Found: '{skill.name}' for goal '{goal[:40]}'")
                        return skill
            except Exception as e:
                logger.debug(f"Error loading skill file {filepath}: {e}")

        return None

    def get_skill_prompt_context(self, goal: str) -> str:
        """
        Generate prompt injection text if a matching learned skill exists.
        """
        skill = self.find_matching_skill(goal)
        if not skill:
            return ""

        lines = [
            f"## 🎓 PREVIOUSLY LEARNED SKILL AVAILABLE: '{skill.name}'",
            f"Description: {skill.description}",
            f"Prerequisites: {', '.join(skill.prerequisites) if skill.prerequisites else 'None'}",
            "Proven Successful Steps from Past Run:",
        ]
        for i, step in enumerate(skill.proven_steps, 1):
            lines.append(f"  Step {i}: {step.get('goal', '')} -> Actions: {step.get('actions', [])}")

        if skill.lessons_learned:
            lines.append("Lessons Learned / Edge Case Advice:")
            for lesson in skill.lessons_learned:
                lines.append(f"  - {lesson}")

        lines.append("RECOMMENDATION: Reuse the proven steps above instead of rediscovering from scratch.")
        return "\n".join(lines)
