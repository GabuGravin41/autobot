"""
Lesson Extractor — Extract cross-run learnings and store in persistent memory.

After each run completes, the lesson extractor analyzes what happened and
stores actionable lessons in the memory store. These lessons appear in the
agent's context on the NEXT run — preventing the agent from repeating the
same mistakes.

Lessons are stored as REMEMBER: entries in the memory store.

Examples:
  - "FAILED 3x on LeetCode: clicking Run Code button at (x=1400, y=700) doesn't work — use keyboard shortcut Ctrl+Enter instead"
  - "ON github.com: DOM clicks are more reliable than coordinate clicks (tested 8x)"
  - "Kaggle notebooks: page takes 15s to load after clicking Run All — always wait(15) after"

This is different from PolicyMemory (which tracks tool success rates numerically) —
Lesson Extractor produces NATURAL LANGUAGE insights that inject directly into the
agent's reasoning context.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class LessonExtractor:
    """
    Analyzes completed run history and extracts actionable lessons.
    Stores them via the MemoryStore for injection into future runs.
    """

    def __init__(self, memory_store: Any) -> None:
        self._memory = memory_store

    def extract_and_store(
        self,
        goal: str,
        history: list[Any],     # list[StepHistoryEntry]
        success: bool,
        run_id: str = "",
    ) -> list[str]:
        """
        Extract lessons from a completed run and store them.

        Returns list of stored lesson keys.
        """
        stored: list[str] = []
        if not history:
            return stored

        try:
            # 1. Repeated failure patterns (same action failed 3+ times on same URL)
            stored += self._extract_failure_lessons(goal, history)

            # 2. Successful tool discoveries (something worked well → remember it)
            stored += self._extract_success_lessons(goal, history)

            # 3. Navigation insights (timing, URL patterns)
            stored += self._extract_nav_lessons(goal, history)

        except Exception as e:
            logger.debug(f"LessonExtractor failed (non-fatal): {e}")

        if stored:
            logger.info(f"📚 LessonExtractor stored {len(stored)} lessons from run {run_id}")
        return stored

    def _extract_failure_lessons(
        self, goal: str, history: list[Any]
    ) -> list[str]:
        """Extract lessons from repeated failures."""
        stored = []
        from collections import Counter

        # Group failures by (url_pattern, action_name)
        failure_groups: dict[str, list[str]] = {}
        for entry in history:
            url = entry.url_before or ""
            url_key = _url_key(url)
            for action, result in zip(entry.agent_output.action, entry.action_results):
                if not result.success and result.error:
                    group_key = f"{url_key}|{action.action_name}"
                    if group_key not in failure_groups:
                        failure_groups[group_key] = []
                    failure_groups[group_key].append(
                        f"{action.action_name}({_action_summary(action)})"
                    )

        # If something failed 3+ times on the same site, store a lesson
        for group_key, failures in failure_groups.items():
            if len(failures) >= 3:
                url_part, action_name = group_key.split("|", 1)
                # Find what eventually worked (if anything)
                worked = self._find_what_worked(history, url_part, action_name)

                lesson = f"On {url_part}: {action_name} failed {len(failures)}x"
                if worked:
                    lesson += f" — instead use: {worked}"

                lesson_key = f"lesson_fail_{_hash_key(url_part + action_name)}"
                self._memory.remember(lesson_key, lesson)
                stored.append(lesson_key)

        return stored

    def _extract_success_lessons(
        self, goal: str, history: list[Any]
    ) -> list[str]:
        """Extract insights from notable successes."""
        stored = []

        # Find steps where agent recovered from prior failures (success after 2+ failures)
        for i, entry in enumerate(history):
            if not any(r.success for r in entry.action_results):
                continue
            # Check if there were failures right before this
            prior_failures = sum(
                1 for e in history[max(0, i-4):i]
                if any(not r.success for r in e.action_results)
            )
            if prior_failures >= 2 and entry.agent_output.action:
                url_key_val = _url_key(entry.url_before or "")
                # What did the agent do that finally worked?
                for action, result in zip(entry.agent_output.action, entry.action_results):
                    if result.success:
                        lesson = (
                            f"On {url_key_val}: after {prior_failures} failures, "
                            f"what worked was {action.action_name}({_action_summary(action)[:60]})"
                        )
                        lesson_key = f"lesson_recover_{_hash_key(url_key_val + str(i))}"
                        self._memory.remember(lesson_key, lesson)
                        stored.append(lesson_key)
                        break  # one lesson per recovery event

        return stored

    def _extract_nav_lessons(
        self, goal: str, history: list[Any]
    ) -> list[str]:
        """Extract navigation timing and URL pattern insights."""
        stored = []

        # Detect pages where the agent needed multiple waits
        wait_counts: dict[str, int] = {}
        for entry in history:
            url = _url_key(entry.url_before or "")
            for action in entry.agent_output.action:
                if action.wait:
                    wait_counts[url] = wait_counts.get(url, 0) + 1

        for url, count in wait_counts.items():
            if count >= 3:
                lesson = f"On {url}: requires {count} wait actions — page loads slowly. Use wait(10)+ after navigating."
                lesson_key = f"lesson_slow_{_hash_key(url)}"
                self._memory.remember(lesson_key, lesson)
                stored.append(lesson_key)

        return stored

    def _find_what_worked(
        self, history: list[Any], url_pattern: str, failed_action: str
    ) -> str | None:
        """Find what action succeeded on a URL where another action kept failing."""
        for entry in history:
            if url_pattern not in _url_key(entry.url_before or ""):
                continue
            for action, result in zip(entry.agent_output.action, entry.action_results):
                if result.success and action.action_name != failed_action:
                    return f"{action.action_name}({_action_summary(action)[:40]})"
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _url_key(url: str) -> str:
    """Extract domain + first path segment."""
    url = re.sub(r"^https?://", "", url.lower().strip())
    url = re.sub(r"^www\.", "", url)
    parts = url.split("/")
    if len(parts) > 1 and parts[1]:
        return f"{parts[0]}/{parts[1][:20]}"
    return parts[0][:30]


def _action_summary(action: Any) -> str:
    """Get a brief summary of an action's parameters."""
    if action.computer_call:
        return action.computer_call.call[:50]
    if action.navigate:
        return action.navigate.url[:40]
    if action.click:
        return f"index={action.click.index}"
    if action.input_text:
        return f"index={action.input_text.index}, text='{action.input_text.text[:20]}'"
    return action.action_name


def _hash_key(s: str) -> str:
    """Short hash for generating unique memory keys."""
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:8]
