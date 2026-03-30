"""
Policy Memory — Maps (context, action_type) → learned preference scores.

This is the agent's "learned intuition" about WHAT WORKS WHERE.

Context = (url_pattern, page_type, task_keywords)
Policy  = {action_tool: {success_rate, avg_reward, confidence}}

The policy starts empty and fills up as the agent runs tasks.
After a few runs on LeetCode, it knows that:
  - "dom.click" has a 90% success rate on leetcode.com/problems
  - "mouse.click" has a 65% success rate there
  → it will suggest preferring DOM clicks over coordinate guessing

The policy is consulted at each step to generate an "affordances" hint
injected into the agent's prompt:
  <affordances>
  On leetcode.com: dom.click (90% success) > mouse.click (65%) > navigate (50%)
  Avoid: keyboard.type on this page type (23% success — try dom.input instead)
  </affordances>

Storage: in-memory with periodic sync to ~/.autobot/policy.json
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path.home() / ".autobot" / "policy.json"


@dataclass
class ActionPolicy:
    """Learned statistics for one (context, action_tool) pair."""
    action_tool: str
    success_count: int = 0
    failure_count: int = 0
    reward_sum: float = 0.0

    @property
    def total(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.5  # prior: assume 50% success for unseen actions
        return self.success_count / self.total

    @property
    def avg_reward(self) -> float:
        if self.total == 0:
            return 0.0
        return self.reward_sum / self.total

    @property
    def confidence(self) -> str:
        """How confident we are in this policy estimate."""
        if self.total >= 15:
            return "high"
        if self.total >= 3:
            return "medium"   # Start showing hints after only 3 observations
        return "low"

    def update(self, success: bool, reward: float) -> None:
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.reward_sum += reward

    def to_dict(self) -> dict:
        return {
            "action_tool": self.action_tool,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "reward_sum": round(self.reward_sum, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionPolicy":
        p = cls(action_tool=d["action_tool"])
        p.success_count = d.get("success_count", 0)
        p.failure_count = d.get("failure_count", 0)
        p.reward_sum = d.get("reward_sum", 0.0)
        return p


class PolicyMemory:
    """
    In-memory policy store with JSON persistence.

    Key = "url_pattern|page_type|task_kw"
    Value = {action_tool: ActionPolicy}
    """

    def __init__(self, path: Path | None = None) -> None:
        env_path = os.getenv("AUTOBOT_POLICY_PATH")
        self.path = Path(env_path) if env_path else (path or _DEFAULT_POLICY_PATH)
        self._lock = threading.Lock()
        # policy[context_key][action_tool] = ActionPolicy
        self._policy: dict[str, dict[str, ActionPolicy]] = defaultdict(dict)
        self._dirty = False
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                with self._lock:
                    for ctx_key, tools in raw.items():
                        self._policy[ctx_key] = {
                            tool: ActionPolicy.from_dict(d)
                            for tool, d in tools.items()
                        }
                logger.debug(f"PolicyMemory: loaded {len(self._policy)} contexts")
        except Exception as e:
            logger.warning(f"PolicyMemory load failed (starting fresh): {e}")

    def save(self) -> None:
        """Persist policy to disk. Call periodically (every N steps)."""
        if not self._dirty:
            return
        try:
            serialized = {}
            with self._lock:
                for ctx_key, tools in self._policy.items():
                    serialized[ctx_key] = {
                        tool: ap.to_dict() for tool, ap in tools.items()
                    }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(serialized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception as e:
            logger.warning(f"PolicyMemory save failed (non-fatal): {e}")

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        url_pattern: str,
        page_type: str,
        task_kw: str,
        action_tool: str,
        success: bool,
        reward: float,
    ) -> None:
        """
        Record the outcome of one action in a given context.
        Called by RLController after each step.
        """
        ctx_key = _make_key(url_pattern, page_type, task_kw)
        with self._lock:
            if action_tool not in self._policy[ctx_key]:
                self._policy[ctx_key][action_tool] = ActionPolicy(action_tool)
            self._policy[ctx_key][action_tool].update(success, reward)
            self._dirty = True

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_preferences(
        self,
        url_pattern: str,
        page_type: str,
        task_kw: str,
        top_k: int = 5,
    ) -> list[ActionPolicy]:
        """
        Return top_k action policies for a context, sorted by avg_reward DESC.

        Only includes policies with medium or high confidence (≥5 observations).
        """
        ctx_key = _make_key(url_pattern, page_type, task_kw)

        # Also try broader context (just url_pattern)
        broad_key = _make_key(url_pattern, "", "")

        policies: dict[str, ActionPolicy] = {}
        with self._lock:
            # Broad context first (less specific but more data)
            for tool, ap in self._policy.get(broad_key, {}).items():
                if ap.total >= 2:
                    policies[tool] = ap
            # Specific context overrides (more data = more weight)
            for tool, ap in self._policy.get(ctx_key, {}).items():
                if ap.total >= 2:
                    policies[tool] = ap

        sorted_policies = sorted(
            policies.values(),
            key=lambda p: (p.avg_reward, p.success_rate),
            reverse=True,
        )
        return sorted_policies[:top_k]

    def build_affordances_hint(
        self,
        url_pattern: str,
        page_type: str,
        task_kw: str,
    ) -> str | None:
        """
        Build a <affordances> hint for the agent prompt.

        Returns None if there is no learned data for this context yet
        (don't inject noise when the policy has no useful information).

        Example output:
            On leetcode.com (coding_challenge page):
            Best tools: dom.click (85%, +0.7 reward, 12 obs) > mouse.click (62%, +0.3)
            ⚠️  Avoid: keyboard.type on this context (31% success)
        """
        prefs = self.get_preferences(url_pattern, page_type, task_kw)
        if not prefs:
            return None

        lines = [f"Learned tool preferences for {url_pattern} ({page_type} page):"]
        good = []
        bad = []
        for ap in prefs:
            if ap.success_rate >= 0.6 and ap.confidence in ("medium", "high"):
                good.append(
                    f"  {ap.action_tool}: {ap.success_rate:.0%} success, "
                    f"{ap.avg_reward:+.1f} reward ({ap.total} obs)"
                )
            elif ap.success_rate < 0.5 and ap.confidence in ("medium", "high"):
                bad.append(
                    f"  {ap.action_tool}: {ap.success_rate:.0%} success — consider alternatives"
                )

        if not good and not bad:
            return None  # all low confidence — don't inject

        if good:
            lines.append("Preferred tools (high historical success):")
            lines.extend(good)
        if bad:
            lines.append("⚠️  Underperforming tools on this context:")
            lines.extend(bad)

        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all learned policies for diagnostics."""
        with self._lock:
            total_contexts = len(self._policy)
            total_observations = sum(
                ap.total
                for tools in self._policy.values()
                for ap in tools.values()
            )
        return {
            "contexts": total_contexts,
            "total_observations": total_observations,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_key(url_pattern: str, page_type: str, task_kw: str) -> str:
    return f"{url_pattern}|{page_type}|{task_kw}"


# ── Wait Duration Memory ───────────────────────────────────────────────────────

_DEFAULT_WAIT_PATH = Path.home() / ".autobot" / "wait_durations.json"
_MAX_SAMPLES = 50       # cap per URL to bound memory usage
_MIN_SAMPLES = 3        # minimum observations before using learned data


class WaitDurationMemory:
    """
    Learns how long things actually take to load/complete per URL pattern.

    Records actual wait durations and returns the 90th-percentile as a
    "safe max" hint so the agent never waits longer than necessary.

    Storage: ~/.autobot/wait_durations.json
    """

    def __init__(self, path: Path | None = None) -> None:
        env_path = os.getenv("AUTOBOT_WAIT_DURATIONS_PATH")
        self.path = Path(env_path) if env_path else (path or _DEFAULT_WAIT_PATH)
        self._lock = threading.Lock()
        # {url_pattern: [duration_seconds, ...]}
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                with self._lock:
                    self._data = {k: v for k, v in raw.items() if isinstance(v, list)}
                logger.debug(f"WaitDurationMemory: loaded {len(self._data)} URL patterns")
        except Exception as e:
            logger.warning(f"WaitDurationMemory load failed (starting fresh): {e}")

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            with self._lock:
                data_copy = {k: v[:] for k, v in self._data.items()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data_copy, indent=2),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception as e:
            logger.warning(f"WaitDurationMemory save failed (non-fatal): {e}")

    def record(self, url_pattern: str, actual_seconds: float) -> None:
        """Record the actual duration it took to wait for this URL pattern."""
        if actual_seconds <= 0:
            return
        with self._lock:
            samples = self._data.setdefault(url_pattern, [])
            samples.append(round(actual_seconds, 2))
            # Keep only the most recent MAX_SAMPLES (sliding window)
            if len(samples) > _MAX_SAMPLES:
                self._data[url_pattern] = samples[-_MAX_SAMPLES:]
            self._dirty = True

    def get_learned_max(self, url_pattern: str) -> float | None:
        """
        Return the 90th-percentile wait duration for this URL pattern.

        Returns None if fewer than MIN_SAMPLES observations exist — caller
        should fall back to the agent's hint_seconds.
        """
        with self._lock:
            samples = self._data.get(url_pattern, [])
        if len(samples) < _MIN_SAMPLES:
            return None
        sorted_samples = sorted(samples)
        idx = int(len(sorted_samples) * 0.90)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "url_patterns": len(self._data),
                "total_observations": sum(len(v) for v in self._data.values()),
            }


# Module-level singletons
policy_memory = PolicyMemory()
wait_duration_memory = WaitDurationMemory()
