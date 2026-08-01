"""
Hierarchical Plan Tree — Structuring complex tasks into nested subplans (plan within a plan within a plan).

Supports:
- Parent-child-grandchild task nodes
- Node status tracking: pending | running | completed | failed | skipped
- Checkpoint serialization to history JSON
- Failure-branching and alternative subplan fallbacks
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

PlanStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass
class PlanNode:
    """A node in the hierarchical plan tree."""

    id: str
    title: str
    description: str
    status: PlanStatus = "pending"
    result_summary: str = ""
    error: str | None = None
    children: list[PlanNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    def add_child(self, node: PlanNode) -> PlanNode:
        """Add a subtask node to this plan node."""
        node.parent_id = self.id
        self.children.append(node)
        return node

    def find_node(self, node_id: str) -> PlanNode | None:
        """Recursively locate a node in the plan tree."""
        if self.id == node_id:
            return self
        for child in self.children:
            found = child.find_node(node_id)
            if found:
                return found
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert plan node tree to dict for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "result_summary": self.result_summary,
            "error": self.error,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanNode:
        """Construct plan node tree from dict."""
        children_data = data.pop("children", [])
        node = cls(**data)
        for child_dict in children_data:
            child_node = cls.from_dict(child_dict)
            child_node.parent_id = node.id
            node.children.append(child_node)
        return node


class HierarchicalPlanTree:
    """
    Manages nested, multi-level plans (Plan within a plan within a plan).
    Provides checkpointing, failure branching, and tree status tracking.
    """

    def __init__(self, root_goal: str) -> None:
        self.root = PlanNode(
            id="node_root",
            title="Root Goal",
            description=root_goal,
        )
        self.active_node_id: str = "node_root"

    def create_subtask(self, parent_id: str, title: str, description: str, metadata: dict | None = None) -> PlanNode:
        """Create a nested subtask under parent_id."""
        parent = self.root.find_node(parent_id) or self.root
        sub_id = f"node_{len(self._all_nodes()) + 1}"
        node = PlanNode(
            id=sub_id,
            title=title,
            description=description,
            metadata=metadata or {},
        )
        parent.add_child(node)
        logger.info(f"🌳 Plan Tree: Created subtask [{sub_id}] '{title}' under [{parent.id}]")
        return node

    def update_status(self, node_id: str, status: PlanStatus, result_summary: str = "", error: str | None = None) -> None:
        """Update node status and error/summary."""
        node = self.root.find_node(node_id)
        if node:
            node.status = status
            if result_summary:
                node.result_summary = result_summary
            if error:
                node.error = error
            logger.info(f"📍 Node [{node_id}] status -> {status}")

    def _all_nodes(self) -> list[PlanNode]:
        """Flat list of all nodes in tree."""
        nodes: list[PlanNode] = []

        def _traverse(n: PlanNode):
            nodes.append(n)
            for c in n.children:
                _traverse(c)

        _traverse(self.root)
        return nodes

    def save_checkpoint(self, filepath: Path) -> None:
        """Save state checkpoint to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "active_node_id": self.active_node_id,
            "tree": self.root.to_dict(),
        }
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def render_ascii_tree(self) -> str:
        """Render human-readable ASCII representation of the plan tree."""
        lines: list[str] = []

        def _render(node: PlanNode, depth: int = 0):
            indent = "  " * depth
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
            }.get(node.status, "❓")
            lines.append(f"{indent}{status_icon} [{node.id}] {node.title} — {node.description[:50]}")
            for child in node.children:
                _render(child, depth + 1)

        _render(self.root)
        return "\n".join(lines)
