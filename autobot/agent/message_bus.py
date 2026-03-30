"""
Message Bus — Async inter-agent communication for multi-agent orchestration.

Provides a lightweight publish/subscribe message bus that:
  - Lets the Orchestrator broadcast task assignments to sub-agents
  - Lets sub-agents report results, status, and errors back
  - Maintains a shared world state (current URL, open apps, clipboard, etc.)
  - Prevents agents from stepping on each other's work

Design: in-process asyncio queues. No external broker needed.
All agents in Autobot run in the same process — a network bus would add
unnecessary latency and complexity for no benefit.

Message format:
{
    "type":      "task" | "result" | "status" | "error" | "world_state",
    "from":      "<agent_id>",
    "to":        "<agent_id>" | "broadcast",
    "payload":   {...},
    "timestamp": float,
}
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Message type constants
MSG_TASK        = "task"
MSG_RESULT      = "result"
MSG_STATUS      = "status"
MSG_ERROR       = "error"
MSG_WORLD_STATE = "world_state"
MSG_CANCEL      = "cancel"


@dataclass
class Message:
    """A single message on the bus."""
    type: str
    from_id: str
    to: str                         # agent_id or "broadcast"
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""

    def __post_init__(self):
        if not self.message_id:
            import uuid
            self.message_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "from": self.from_id,
            "to": self.to,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }


class AgentInbox:
    """An asyncio queue with convenience methods for a single agent."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=50)

    async def put(self, msg: Message) -> None:
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            logger.warning(f"Inbox full for {self.agent_id} — dropping message {msg.message_id}")

    async def get(self, timeout: float = 30.0) -> Message | None:
        """Wait up to timeout seconds for a message. Returns None on timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_nowait(self) -> Message | None:
        """Non-blocking get. Returns None if queue is empty."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @property
    def pending(self) -> int:
        return self._queue.qsize()


class WorldState:
    """
    Shared mutable state visible to all agents.

    Updated by agents as they observe and change the environment.
    The Orchestrator uses this to make routing decisions.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "current_url": "",
            "active_app": "browser",
            "open_tabs": [],
            "clipboard": "",
            "last_action": "",
            "last_action_success": None,
            "current_agent": "",
            "step_number": 0,
        }
        self._lock = asyncio.Lock()

    async def update(self, **kwargs) -> None:
        async with self._lock:
            self._state.update(kwargs)
            self._state["updated_at"] = time.time()

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._state.get(key, default)

    async def snapshot(self) -> dict:
        async with self._lock:
            return dict(self._state)

    def get_sync(self, key: str, default: Any = None) -> Any:
        """Non-async read for use in non-async contexts."""
        return self._state.get(key, default)


class MessageBus:
    """
    Central message broker for multi-agent communication.

    Usage:
        bus = MessageBus()

        # Register agents
        bus.register("orchestrator")
        bus.register("web_navigator")

        # Send a message
        await bus.send(Message(type=MSG_TASK, from_id="orchestrator",
                               to="web_navigator", payload={"task": "..."}))

        # Agent reads its inbox
        msg = await bus.inbox("web_navigator").get(timeout=5.0)
    """

    def __init__(self) -> None:
        self._inboxes: dict[str, AgentInbox] = {}
        self._history: list[Message] = []
        self._max_history = 500
        self.world_state = WorldState()

    def register(self, agent_id: str) -> AgentInbox:
        """Register an agent and create its inbox. Returns the inbox."""
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = AgentInbox(agent_id)
            logger.debug(f"MessageBus: registered agent {agent_id!r}")
        return self._inboxes[agent_id]

    def inbox(self, agent_id: str) -> AgentInbox:
        """Get an agent's inbox (creates it if not registered)."""
        return self.register(agent_id)

    async def send(self, msg: Message) -> None:
        """
        Deliver a message to its target.

        If to="broadcast", delivers to ALL registered agents except the sender.
        """
        # Archive in history
        self._history.append(msg)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        if msg.to == "broadcast":
            for agent_id, inbox in self._inboxes.items():
                if agent_id != msg.from_id:
                    await inbox.put(msg)
        else:
            inbox = self._inboxes.get(msg.to)
            if inbox:
                await inbox.put(msg)
            else:
                logger.warning(
                    f"MessageBus: no inbox for {msg.to!r} — "
                    f"message {msg.message_id} dropped"
                )

    async def broadcast(self, from_id: str, msg_type: str, payload: dict) -> None:
        """Convenience: broadcast a message to all agents."""
        await self.send(Message(type=msg_type, from_id=from_id, to="broadcast", payload=payload))

    def get_history(
        self, agent_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Return recent message history, optionally filtered by agent."""
        msgs = self._history
        if agent_id:
            msgs = [m for m in msgs if m.from_id == agent_id or m.to in (agent_id, "broadcast")]
        return [m.to_dict() for m in msgs[-limit:]]

    def registered_agents(self) -> list[str]:
        return list(self._inboxes.keys())


# Module-level singleton
message_bus = MessageBus()
