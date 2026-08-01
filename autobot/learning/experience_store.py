"""
Experience Store — SQLite-backed storage for agent (state, action, outcome) tuples.

Each "experience" is one step the agent took:
  - state:    what was observed (url_pattern, page_type, task_keywords)
  - action:   what the agent did (action_name, action_params_hash)
  - outcome:  what happened (success/failure, error_type, reward)
  - context:  task goal keywords

Over time the store accumulates thousands of experiences that the
PolicyMemory uses to compute action preference scores.

Storage: ~/.autobot/experiences.db (or AUTOBOT_EXPERIENCES_PATH env var)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".autobot" / "experiences.db"


@dataclass
class Experience:
    """A single recorded (state, action, outcome) tuple."""
    # State features
    url_pattern: str         # e.g. "github.com", "leetcode.com/problems"
    page_type: str           # e.g. "form", "list", "code_editor", "error"
    task_keywords: str       # first 5 words of the goal, normalised

    # Action
    action_name: str         # e.g. "computer_call", "navigate", "click"
    action_tool: str         # e.g. "mouse.click", "keyboard.type", "dom_click"
    action_context: str      # hash of params (not stored literally to save space)

    # Outcome
    success: bool
    error_type: str          # e.g. "timeout", "not_found", "permission", ""
    reward: float            # scalar reward signal

    # Metadata
    step_number: int
    run_id: str
    timestamp: str           # ISO 8601


@dataclass
class StepState:
    """Lightweight state snapshot passed to ExperienceStore after each step."""
    url: str
    goal: str
    action_name: str
    action_params: dict
    success: bool
    error: str | None
    step_number: int
    run_id: str


class ExperienceStore:
    """
    Persistent SQLite store for agent experiences.

    Thread-safe: uses a single write lock for INSERT/UPDATE operations.
    Reads do not acquire the lock (SQLite allows concurrent readers).
    """

    def __init__(self, path: Path | None = None) -> None:
        env_path = os.getenv("AUTOBOT_EXPERIENCES_PATH")
        self.path = Path(env_path) if env_path else (path or _DEFAULT_PATH)
        self._lock = threading.Lock()
        self._init_db()

    # ── Schema ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the experiences table if it doesn't exist."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    url_pattern TEXT NOT NULL,
                    page_type   TEXT NOT NULL,
                    task_kw     TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    action_tool TEXT NOT NULL,
                    action_ctx  TEXT NOT NULL,
                    success     INTEGER NOT NULL,
                    error_type  TEXT NOT NULL DEFAULT '',
                    reward      REAL NOT NULL DEFAULT 0.0,
                    step_number INTEGER NOT NULL DEFAULT 0,
                    run_id      TEXT NOT NULL DEFAULT '',
                    ts          TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context
                ON experiences(url_pattern, action_name, action_tool)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task
                ON experiences(task_kw, action_name)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"ExperienceStore init failed (non-fatal): {e}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, state: StepState, reward: float) -> None:
        """
        Record one experience tuple.  Called by RLController after each step.
        Non-blocking: failures are logged but never propagated to the caller.
        """
        try:
            url_pattern = _url_pattern(state.url)
            page_type = _infer_page_type(state.url, state.action_params)
            task_kw = _task_keywords(state.goal)
            action_tool = _action_tool(state.action_name, state.action_params)
            action_ctx = _param_hash(state.action_params)
            error_type = _classify_error(state.error)
            ts = datetime.now(timezone.utc).isoformat()

            with self._lock:
                conn = sqlite3.connect(str(self.path))
                conn.execute(
                    """
                    INSERT INTO experiences
                      (url_pattern, page_type, task_kw,
                       action_name, action_tool, action_ctx,
                       success, error_type, reward,
                       step_number, run_id, ts)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        url_pattern, page_type, task_kw,
                        state.action_name, action_tool, action_ctx,
                        int(state.success), error_type, reward,
                        state.step_number, state.run_id, ts,
                    ),
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.debug(f"Experience record failed (non-fatal): {e}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def query_success_rates(
        self,
        url_pattern: str,
        action_names: list[str] | None = None,
        min_observations: int = 3,
    ) -> dict[str, dict[str, Any]]:
        """
        Return success rate and average reward per action_tool for a given URL pattern.

        Result format:
        {
            "mouse.click":     {"success_rate": 0.72, "avg_reward": 0.4, "count": 25},
            "keyboard.type":   {"success_rate": 0.91, "avg_reward": 0.8, "count": 11},
            ...
        }
        """
        try:
            conn = sqlite3.connect(str(self.path))
            conn.row_factory = sqlite3.Row

            filters = ["url_pattern LIKE ?"]
            params: list[Any] = [f"%{url_pattern}%"]

            if action_names:
                placeholders = ",".join("?" * len(action_names))
                filters.append(f"action_name IN ({placeholders})")
                params.extend(action_names)

            where = " AND ".join(filters)
            rows = conn.execute(
                f"""
                SELECT action_tool,
                       COUNT(*) as cnt,
                       AVG(success) as sr,
                       AVG(reward)  as ar
                FROM experiences
                WHERE {where}
                GROUP BY action_tool
                HAVING COUNT(*) >= ?
                ORDER BY ar DESC
                """,
                params + [min_observations],
            ).fetchall()
            conn.close()

            return {
                row["action_tool"]: {
                    "success_rate": round(row["sr"], 3),
                    "avg_reward": round(row["ar"], 3),
                    "count": row["cnt"],
                }
                for row in rows
            }
        except Exception as e:
            logger.debug(f"ExperienceStore query failed: {e}")
            return {}

    def query_error_patterns(
        self,
        url_pattern: str,
        action_name: str,
        limit: int = 5,
    ) -> list[str]:
        """Return the most common error types for a (url, action) pair."""
        try:
            conn = sqlite3.connect(str(self.path))
            rows = conn.execute(
                """
                SELECT error_type, COUNT(*) as cnt
                FROM experiences
                WHERE url_pattern LIKE ? AND action_name = ? AND error_type != ''
                GROUP BY error_type
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (f"%{url_pattern}%", action_name, limit),
            ).fetchall()
            conn.close()
            return [r[0] for r in rows if r[0]]
        except Exception:
            return []

    def recent_run_summary(self, run_id: str) -> dict[str, Any]:
        """Return aggregate stats for a specific run."""
        try:
            conn = sqlite3.connect(str(self.path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT COUNT(*) as total,
                       SUM(success) as successes,
                       AVG(reward) as avg_reward,
                       MAX(step_number) as max_step
                FROM experiences
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            conn.close()
            if not row:
                return {}
            return dict(row)
        except Exception:
            return {}

    def total_experiences(self) -> int:
        """Return total number of recorded experiences."""
        try:
            conn = sqlite3.connect(str(self.path))
            n = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            conn.close()
            return n
        except Exception:
            return 0


# ── Utility helpers ───────────────────────────────────────────────────────────

def _url_pattern(url: str) -> str:
    """Extract the domain + first path segment from a URL as a pattern key."""
    url = url.strip().lower()
    # Strip protocol
    url = re.sub(r"^https?://", "", url)
    # Keep domain + first path segment (max 40 chars)
    parts = url.split("/")
    domain = parts[0].replace("www.", "")
    if len(parts) > 1 and parts[1]:
        return f"{domain}/{parts[1][:20]}"
    return domain[:40]


def _infer_page_type(url: str, action_params: dict) -> str:
    """
    Heuristically classify the page type from URL and action params.

    Used as a context key in PolicyMemory so the agent learns that
    e.g. "form_input" pages favor keyboard.type over mouse.click.
    """
    url_lower = url.lower()

    # Auth / login
    if any(k in url_lower for k in ["login", "signin", "sign-in", "auth", "authenticate", "oauth", "sso"]):
        return "auth"
    # Registration
    if any(k in url_lower for k in ["register", "signup", "sign-up", "create-account"]):
        return "registration"
    # Code repos and editors
    if any(k in url_lower for k in ["github.com", "gitlab.com", "bitbucket.org"]):
        return "code_repo"
    # Coding challenges
    if any(k in url_lower for k in ["leetcode", "codeforces", "hackerrank", "codewars", "atcoder"]):
        return "coding_challenge"
    # Data platforms
    if any(k in url_lower for k in ["kaggle.com", "huggingface.co", "colab.research"]):
        return "data_platform"
    # Search pages
    if any(k in url_lower for k in ["google.com/search", "bing.com/search", "duckduckgo.com", "search?q=", "results"]):
        return "search"
    # Error pages
    if any(k in url_lower for k in ["404", "error", "not-found", "not_found"]):
        return "error"
    # Docs / documentation pages
    if any(k in url_lower for k in ["/docs/", "/documentation/", "readthedocs", "developer.mozilla"]):
        return "docs"
    # Dashboard / admin panels
    if any(k in url_lower for k in ["/dashboard", "/admin", "/settings", "/account", "/profile"]):
        return "dashboard"
    # Action-based hints
    call = action_params.get("call", "")
    if "keyboard.type" in call or "dom.input" in call:
        return "form_input"
    if "navigate" in str(action_params):
        return "navigation"
    return "general"


def _task_keywords(goal: str) -> str:
    """
    Extract the top meaningful keywords from a goal for context matching.

    Prioritizes domain nouns over generic verbs. Returns up to 5 keywords
    joined by spaces — used as a PolicyMemory context dimension.
    """
    words = re.findall(r"\b[a-zA-Z]{3,}\b", goal.lower())
    # Extended stop list — remove generic action verbs and articles
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "are", "has",
        "was", "find", "open", "click", "search", "use", "get", "make",
        "create", "add", "run", "start", "then", "into", "your", "will",
        "you", "can", "all", "any", "new", "go", "set", "try", "its",
        "take", "put", "let", "see",
    }
    # Boost domain-specific keywords that appear in common task sites
    _DOMAIN_BOOST = {
        "leetcode", "kaggle", "github", "google", "youtube", "arxiv",
        "python", "javascript", "typescript", "notebook", "dataset",
        "competition", "submission", "solution", "problem", "article",
    }
    boosted = [w for w in words if w in _DOMAIN_BOOST]
    regular = [w for w in words if w not in stop and w not in _DOMAIN_BOOST]
    # Combine: domain words first, then regular keywords, cap at 5
    combined = (boosted + regular)[:5]
    return " ".join(combined)


def _action_tool(action_name: str, params: dict) -> str:
    """
    Extract the specific tool used from the action.

    Maps action_name → granular tool label:
      "computer_call" with call="computer.mouse.click(...)"  → "mouse.click"
      "navigate"                                              → "navigate"
      "click"                                                 → "dom.click"
    """
    if action_name == "computer_call":
        call = params.get("call", "")
        m = re.match(r"computer\.(\w+\.\w+)", call)
        if m:
            return m.group(1)
        return "computer_call"
    if action_name == "click":
        return "dom.click"
    if action_name == "input_text":
        return "dom.input"
    if action_name in ("press_key",):
        return "keyboard.key"
    return action_name


def _param_hash(params: dict) -> str:
    """Short hash of action parameters — used for context fingerprinting."""
    try:
        serialized = json.dumps(params, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _classify_error(error: str | None) -> str:
    """Classify an error string into a category."""
    if not error:
        return ""
    e = error.lower()
    if any(k in e for k in ["timeout", "timed out", "time out"]):
        return "timeout"
    if any(k in e for k in ["not found", "does not exist", "no element"]):
        return "not_found"
    if any(k in e for k in ["permission", "denied", "forbidden", "unauthorized"]):
        return "permission"
    if any(k in e for k in ["click failed", "element not clickable", "intercepted"]):
        return "click_failed"
    if any(k in e for k in ["network", "connection", "dns"]):
        return "network"
    return "other"


# Module-level singleton
experience_store = ExperienceStore()
