"""
HumanGate — Shared registry for approval requests that pause the agent.

The agent loop calls `wait_for_approval()` which registers a pending request
and blocks until the user responds via the API (or times out).

The API layer calls `respond()` to unblock the agent.

Design:
  - Pure asyncio, no threading
  - All coroutines run in the same FastAPI event loop
  - No circular imports — app.py and approval.py both import from here

Usage in agent:
    allowed = await human_gate.wait_for_approval(
        key="del-file-abc",
        message="Agent wants to delete /home/user/report.docx",
        timeout=120,
    )

Usage in API:
    from autobot.agent.human_gate import get_pending, respond
    pending = get_pending()      # → {"key": ..., "message": ...} | None
    respond("del-file-abc", "allow")
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# key → asyncio.Event
_events: dict[str, asyncio.Event] = {}

# key → "allow" | "block"
_responses: dict[str, str] = {}

# key → human-readable message shown in the UI
_messages: dict[str, str] = {}

# ── User → Agent message queue ────────────────────────────────────────────────
# User can inject free-form instructions into the running agent at any time.
# The agent drains this queue at the start of each step.
_user_message_queue: list[dict[str, Any]] = []

# ── Agent → User messages (narrative / questions) ────────────────────────────
# The agent can surface thoughts or questions; the frontend polls these.
_agent_messages: list[dict[str, Any]] = []


def inject_user_message(text: str) -> None:
    """Inject a user instruction into the running agent's next step."""
    import time
    _user_message_queue.append({"text": text, "ts": time.time()})
    logger.info(f"[HumanGate] User message queued: {text[:100]}")


def pop_user_messages() -> list[str]:
    """Drain and return all queued user messages (called by agent at step start)."""
    msgs = [m["text"] for m in _user_message_queue]
    _user_message_queue.clear()
    return msgs


def push_agent_message(text: str, kind: str = "narrative") -> None:
    """Agent surfaces a thought or question to the user."""
    import time
    _agent_messages.append({"text": text, "kind": kind, "ts": time.time()})
    # Keep only the last 50 to avoid unbounded growth
    if len(_agent_messages) > 50:
        _agent_messages.pop(0)


def get_agent_messages(since_ts: float = 0.0) -> list[dict[str, Any]]:
    """Return agent messages newer than since_ts for the frontend to display."""
    return [m for m in _agent_messages if m["ts"] > since_ts]


async def wait_for_approval(
    key: str,
    message: str,
    timeout: float = 120.0,
) -> bool:
    """
    Pause the agent and wait for the user to approve or block an action.

    Returns True if the user clicked "Allow", False on "Block" or timeout.
    """
    event = asyncio.Event()
    _events[key] = event
    _messages[key] = message
    logger.info(f"[HumanGate] Waiting for approval: {key} — {message[:100]}")

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        response = _responses.get(key, "block")
        allowed = response == "allow"
        logger.info(f"[HumanGate] {key} → {response}")
        return allowed
    except asyncio.TimeoutError:
        logger.warning(f"[HumanGate] {key} timed out after {timeout}s — defaulting to block")
        return False
    finally:
        _events.pop(key, None)
        _responses.pop(key, None)
        _messages.pop(key, None)


def respond(key: str, response: str) -> bool:
    """
    Respond to a pending approval request.

    Args:
        key:      The approval key (must match what the agent registered).
        response: "allow" or "block".

    Returns True if the key was found and the event was fired.
    """
    if key not in _events:
        logger.warning(f"[HumanGate] respond() called for unknown key: {key}")
        return False
    _responses[key] = response
    _events[key].set()
    return True


def get_pending() -> dict[str, Any] | None:
    """
    Return the first pending (unanswered) approval request, or None.

    The API exposes this so the frontend knows what to show.
    """
    for key, message in _messages.items():
        if key in _events and not _events[key].is_set():
            return {"key": key, "message": message}
    return None


def get_all_pending() -> list[dict[str, Any]]:
    """Return all unanswered requests (in case multiple stack up)."""
    return [
        {"key": k, "message": m}
        for k, m in _messages.items()
        if k in _events and not _events[k].is_set()
    ]
