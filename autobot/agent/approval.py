"""
ApprovalGuard — Risk-based action gating for autonomous agent runs.

Three modes (set via AUTOBOT_APPROVAL_MODE env var or settings API):
  strict   — Ask user before ANY action in the CAUTION or DANGER tier
  balanced — Ask user only for DANGER-tier actions (default)
  trusted  — Never interrupt; run everything automatically

Risk tiers:
  SAFE    — Navigation, clicks, typing, scrolling, screenshots, reading
  CAUTION — Sending messages/emails, form submissions, file uploads
  DANGER  — Deleting files, making purchases, executing shell commands,
             writing to system directories, clearing data

Usage (from AgentLoop._execute_step):
    from autobot.agent.approval import ApprovalGuard, RiskTier
    guard = ApprovalGuard(mode="balanced")
    risk = guard.classify(action)
    if not await guard.gate(action, risk, goal=agent_output.next_goal):
        return  # user blocked — skip this action
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from autobot.agent.models import ActionModel


class RiskTier(str, Enum):
    SAFE    = "safe"
    CAUTION = "caution"
    DANGER  = "danger"


# ── Keyword sets for risk classification ──────────────────────────────────────

_DANGER_PATTERNS = [
    # File deletion
    r"\brm\b", r"\brmdir\b", r"shutil\.rmtree", r"os\.remove", r"os\.unlink",
    r"\.delete\(", r"\.remove\(",
    # Shell execution that can be destructive
    r"subprocess", r"os\.system", r"shell=True",
    # Payment / purchase
    r"\bpurchase\b", r"\bpay\b", r"\bcheckout\b", r"\bbuy now\b",
    r"credit.?card", r"payment", r"stripe", r"paypal",
    # Database destruction
    r"\bdrop\b.*\btable\b", r"\btruncate\b", r"\bdelete from\b",
    # Disk-level
    r"\bformat\b", r"\bmkfs\b", r"\bdd\b.*\bof=\b",
    # Account-level
    r"delete.?account", r"close.?account", r"deactivate",
]

_CAUTION_PATTERNS = [
    # Sending messages
    r"\bsend\b", r"\bsubmit\b", r"\.send\(", r"\.submit\(",
    r"\bemail\b", r"\bslack\b", r"\bwhatsapp\b", r"\btelegram\b",
    r"\btweet\b", r"\bpost\b",
    # File write (to home/docs but not temp)
    r"open\(.*['\"]w['\"]", r"\.write\(",
    # Git push / publish
    r"\bgit push\b", r"\bgit commit\b", r"\bnpm publish\b",
    # Form submission keywords in UI
    r"confirm", r"agree", r"accept", r"approve",
]

_DANGER_RE = re.compile("|".join(_DANGER_PATTERNS), re.IGNORECASE)
_CAUTION_RE = re.compile("|".join(_CAUTION_PATTERNS), re.IGNORECASE)


def _action_text(action: "ActionModel") -> str:
    """Extract the human-readable text of an action for risk classification."""
    parts: list[str] = []
    if action.computer_call:
        parts.append(action.computer_call.call)
    if action.navigate:
        parts.append(f"navigate to {action.navigate.url}")
    if action.input_text:
        parts.append(f"type: {action.input_text.text}")
    if action.input_text_native:
        parts.append(f"type: {action.input_text_native.text}")
    if action.done:
        parts.append(f"done: {action.done.text}")
    return " | ".join(parts) or action.action_name


# ── ApprovalGuard ─────────────────────────────────────────────────────────────

class ApprovalGuard:
    """
    Classifies each agent action by risk and gates it against the
    user's chosen approval mode.

    Instantiated once per AgentLoop run; mode is read from env at creation time
    so it can be changed between runs via the settings API.
    """

    def __init__(self, mode: str | None = None) -> None:
        self.mode = (mode or os.getenv("AUTOBOT_APPROVAL_MODE", "balanced")).lower()
        logger.info(f"ApprovalGuard active — mode: {self.mode}")

    def classify(self, action: "ActionModel") -> RiskTier:
        """Classify an action into SAFE / CAUTION / DANGER."""
        text = _action_text(action)
        if _DANGER_RE.search(text):
            return RiskTier.DANGER
        if _CAUTION_RE.search(text):
            return RiskTier.CAUTION
        return RiskTier.SAFE

    def needs_approval(self, tier: RiskTier) -> bool:
        """Does this risk tier require user approval under the current mode?"""
        if self.mode == "trusted":
            return False
        if self.mode == "strict":
            return tier in (RiskTier.CAUTION, RiskTier.DANGER)
        # balanced (default)
        return tier == RiskTier.DANGER

    async def gate(
        self,
        action: "ActionModel",
        tier: RiskTier,
        goal: str = "",
        timeout: float = 120.0,
    ) -> bool:
        """
        Gate an action.

        Returns True if the action should proceed, False if it should be skipped.
        In trusted/safe modes this returns immediately without blocking.
        """
        if not self.needs_approval(tier):
            return True  # proceed immediately

        from autobot.agent.human_gate import wait_for_approval

        text = _action_text(action)
        # Stable key so the same action in a tight loop produces the same key
        key = "approval_" + hashlib.md5(text.encode()).hexdigest()[:10]
        tier_label = tier.value.upper()
        message = (
            f"[{tier_label}] Agent wants to: {text[:200]}"
            + (f"\n\nContext: {goal[:200]}" if goal else "")
        )

        logger.warning(f"⛔ Approval required ({self.mode} mode, {tier_label}): {text[:100]}")
        allowed = await wait_for_approval(key=key, message=message, timeout=timeout)

        if allowed:
            logger.info(f"✅ User approved: {text[:80]}")
        else:
            logger.warning(f"🚫 User blocked (or timed out): {text[:80]}")

        return allowed
