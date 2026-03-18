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

    def recall(self, query: str, top_k: int = 8) -> list[tuple[str, str]]:
        """
        Return up to top_k (key, value) pairs most relevant to query.
        Relevance = number of query words that appear in key or value.
        """
        if not self._data:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return []

        scored: list[tuple[int, str]] = []
        for key, entry in self._data.items():
            haystack = f"{key} {entry['value']}".lower()
            score = sum(1 for w in query_words if w in haystack)
            if score > 0:
                scored.append((score, key))

        scored.sort(reverse=True)
        results = []
        with self._lock:
            for _, key in scored[:top_k]:
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

    def __len__(self) -> int:
        return len(self._data)


def _normalise_key(key: str) -> str:
    """'My Kaggle Username' → 'my_kaggle_username'"""
    return re.sub(r"\s+", "_", key.strip().lower())


# Module-level singleton — shared across the process
memory_store = MemoryStore()
