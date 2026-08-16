"""
Permission Engine — Manages levels of autonomy and irreversible action evaluation.
"""
from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PermissionLevel(str, enum.Enum):
    """Levels of AI Autonomy on the laptop."""
    LEVEL_0_OBSERVER = "level_0_observer"        # Read-only advice, no execution allowed
    LEVEL_1_SUPERVISED = "level_1_supervised"    # Auto-approve read actions; ask before irreversible actions
    LEVEL_2_FULL_AUTONOMY = "level_2_full"       # Full autonomous execution


IRREVERSIBLE_KEYWORDS = {
    "delete", "remove", "rm", "format", "send", "submit",
    "pay", "transfer", "buy", "purchase", "kill", "shutdown",
    "restart", "wipe", "drop", "truncate"
}


@dataclass
class PermissionCheckResult:
    allowed: bool
    requires_user_approval: bool
    reason: str


class PermissionManager:
    """Evaluates agent actions against the user's permission dial."""

    def __init__(self, level: PermissionLevel = PermissionLevel.LEVEL_1_SUPERVISED):
        env_level = os.getenv("AUTOBOT_PERMISSION_LEVEL", "").lower()
        if env_level in {p.value for p in PermissionLevel}:
            self.level = PermissionLevel(env_level)
        elif os.getenv("AUTOBOT_AUTO_APPROVE") == "1":
            self.level = PermissionLevel.LEVEL_2_FULL_AUTONOMY
        else:
            self.level = level

    def check_action(self, action_name: str, args: Dict[str, Any]) -> PermissionCheckResult:
        """Check if an action is permitted under the current permission dial."""
        if self.level == PermissionLevel.LEVEL_0_OBSERVER:
            return PermissionCheckResult(
                allowed=False,
                requires_user_approval=True,
                reason="System is in Level 0 (Observer Mode). Actions require manual user confirmation."
            )

        if self.level == PermissionLevel.LEVEL_2_FULL_AUTONOMY:
            return PermissionCheckResult(
                allowed=True,
                requires_user_approval=False,
                reason="Full Autonomy Mode enabled."
            )

        # Level 1: Supervised mode — evaluate for irreversible actions
        is_irreversible = self._is_irreversible(action_name, args)
        if is_irreversible:
            return PermissionCheckResult(
                allowed=False,
                requires_user_approval=True,
                reason=f"Action '{action_name}' is potentially irreversible and requires user approval."
            )

        return PermissionCheckResult(
            allowed=True,
            requires_user_approval=False,
            reason="Safe action auto-approved under Supervised Mode."
        )

    def _is_irreversible(self, action_name: str, args: Dict[str, Any]) -> bool:
        """Heuristic check for irreversible OS / web operations."""
        arg_str = str(args).lower()
        if any(kw in action_name.lower() for kw in IRREVERSIBLE_KEYWORDS):
            return True
        if any(kw in arg_str for kw in IRREVERSIBLE_KEYWORDS):
            return True
        return False
