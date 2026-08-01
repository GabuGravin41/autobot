"""
Skill Distiller — Distills successful execution trajectories into reusable, learned skills.

Prevents Autobot from doing blind searching on tasks it has already solved before.
When a task completes successfully, SkillDistiller extracts proven action sequences,
lessons learned, and edge-case fallbacks, saving them into persistent skill files.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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
        safe_name = skill.name.lower().replace(" ", "_").replace("/", "_")
        filepath = self.skills_dir / f"{safe_name}.json"
        filepath.write_text(json.dumps(skill.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"🎓 Learned Skill Saved: '{skill.name}' at '{filepath}'")
        return filepath

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
