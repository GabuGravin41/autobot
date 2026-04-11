"""
Persistent Memory Store — cross-run fact storage for Autobot.

The agent writes "REMEMBER:key=value" in its memory field; the loop
parses it and calls memory_store.remember(). On every step, the loop
calls memory_store.recall(goal) to inject relevant facts into the prompt.

Storage: ~/.autobot/memory.json  (or AUTOBOT_MEMORY_PATH env var)
Format:  {"key": {"value": "...", "updated": "ISO-timestamp", "hits": N}}

Recall strategy: keyword overlap between query and stored keys/values.
No vector DB needed — fast enough for thousands of entries.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".autobot" / "memory.json"


class MemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        env_path = os.getenv("AUTOBOT_MEMORY_PATH")
        self.path = Path(env_path) if env_path else (path or _DEFAULT_PATH)
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Memory load failed (starting fresh): {e}")
            self._data = {}

    def _save(self) -> None:
        # Caller must hold self._lock
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Memory save failed (non-fatal): {e}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(self, key: str, value: str) -> None:
        """Store or update a fact. Key is normalised to lowercase-underscore."""
        key = _normalise_key(key)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._data.get(key)
            self._data[key] = {
                "value": value.strip(),
                "updated": now,
                "hits": (existing["hits"] if existing else 0),
            }
            self._save()
        logger.info(f"🧠 Remembered: {key} = {value[:60]}")

    def forget(self, key: str) -> None:
        key = _normalise_key(key)
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._save()

    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 4) -> list[tuple[str, str]]:
        """
        Return up to top_k (key, value) pairs most relevant to query.
        Relevance = number of query words that appear in key or value.
        Tie-broken by hit count (most-used memories preferred).
        """
        if not self._data:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return []

        scored: list[tuple[int, int, str]] = []
        for key, entry in self._data.items():
            haystack = f"{key} {entry['value']}".lower()
            score = sum(1 for w in query_words if w in haystack)
            if score > 0:
                hits = entry.get("hits", 0)
                scored.append((score, hits, key))

        # Sort by relevance score DESC, then hits DESC (most-used memories break ties)
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        results = []
        with self._lock:
            for _, _hits, key in scored[:top_k]:
                entry = self._data[key]
                entry["hits"] = entry.get("hits", 0) + 1
                results.append((key, entry["value"]))
            if results:
                self._save()  # persist hit counts
        return results

    def all_entries(self) -> list[tuple[str, str]]:
        """Return all stored facts as (key, value) pairs, newest first."""
        sorted_items = sorted(
            self._data.items(),
            key=lambda x: x[1].get("updated", ""),
            reverse=True,
        )
        return [(k, v["value"]) for k, v in sorted_items]

    def prune(
        self,
        max_age_days: int = 60,
        max_entries: int = 500,
        min_hits_for_old: int = 1,
    ) -> int:
        """
        Remove stale or low-value entries to keep memory lean.

        Pruning rules (applied in order):
          1. Entries older than max_age_days AND never recalled (hits=0) → remove
          2. If still over max_entries, remove oldest zero-hit entries first,
             then oldest low-hit entries, until under the limit.

        Returns the number of entries removed.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        removed = 0

        with self._lock:
            # Rule 1: prune old zero-hit entries
            stale = [
                k for k, v in self._data.items()
                if v.get("hits", 0) < min_hits_for_old
                and _parse_dt(v.get("updated", "")) < cutoff
            ]
            for k in stale:
                del self._data[k]
                removed += 1

            # Rule 2: enforce max_entries cap
            if len(self._data) > max_entries:
                # Sort by (hits asc, updated asc) — remove least useful entries first
                sortable = sorted(
                    self._data.items(),
                    key=lambda x: (x[1].get("hits", 0), x[1].get("updated", "")),
                )
                excess = len(self._data) - max_entries
                for k, _ in sortable[:excess]:
                    del self._data[k]
                    removed += 1

            if removed:
                self._save()

        if removed:
            logger.info(f"🧹 Memory pruned: {removed} stale entries removed ({len(self._data)} remain)")
        return removed

    def stats(self) -> dict:
        """Return memory statistics."""
        hits = [v.get("hits", 0) for v in self._data.values()]
        return {
            "total": len(self._data),
            "total_hits": sum(hits),
            "zero_hit_entries": sum(1 for h in hits if h == 0),
            "high_value_entries": sum(1 for h in hits if h >= 5),
        }

    def __len__(self) -> int:
        return len(self._data)


def _normalise_key(key: str) -> str:
    """'My Kaggle Username' → 'my_kaggle_username'"""
    return re.sub(r"\s+", "_", key.strip().lower())


def _parse_dt(iso: str) -> datetime:
    """Parse ISO timestamp string, returning epoch on failure."""
    try:
        return datetime.fromisoformat(iso)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


# Module-level singleton — shared across the process
memory_store = MemoryStore()
# Prune stale entries on startup (silent, non-fatal)
try:
    memory_store.prune(max_age_days=60, max_entries=500)
except Exception:
    pass
