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

    async def gate(
        self,
        action: "ActionModel",
        tier: RiskTier,
        goal: str = "",
        timeout: float = 120.0,
    ) -> bool:
        """
        Gate an action based on current approval mode and risk tier.

        trusted  — Always proceeds. Sends a desktop notification for DANGER actions
                   so the user is informed but the agent never pauses.
        balanced — Pauses for DANGER only; CAUTION proceeds with a warning log.
        strict   — Pauses for CAUTION and DANGER; requires explicit Allow from user.

        Returns True to proceed, False to skip this action.
        """
        text = _action_text(action)
        tier_label = tier.value.upper()

        if tier == RiskTier.SAFE:
            return True

        if self.mode == "trusted":
            if tier == RiskTier.DANGER:
                logger.warning(f"[TRUSTED/DANGER] Proceeding without pause: {text[:100]}")
                _send_notification(
                    title="Autobot — Risky Action (trusted mode)",
                    body=f"Doing: {text[:120]}\nYou can pause or abort from the Autobot dashboard.",
                )
            return True  # trusted never pauses

        if self.mode == "balanced" and tier == RiskTier.CAUTION:
            logger.info(f"[BALANCED/CAUTION] Proceeding: {text[:100]}")
            return True  # balanced only pauses for DANGER

        # strict: pause for CAUTION + DANGER
        # balanced: pause for DANGER
        from autobot.agent.human_gate import wait_for_approval

        key = "approval_" + hashlib.md5(text.encode()).hexdigest()[:10]
        message = (
            f"[{tier_label}] Agent wants to:\n{text[:200]}"
            + (f"\n\nCurrent goal: {goal[:200]}" if goal else "")
            + f"\n\nMode: {self.mode} — click Allow to proceed or Block to skip."
        )
        logger.warning(f"⛔ Approval required ({self.mode}/{tier_label}): {text[:100]}")
        _send_notification(
            title=f"Autobot needs your approval ({tier_label})",
            body=f"{text[:120]}\nOpen the Autobot dashboard to Allow or Block.",
        )
        allowed = await wait_for_approval(key=key, message=message, timeout=timeout)
        logger.info(f"{'✅ Approved' if allowed else '🚫 Blocked'}: {text[:80]}")
        return allowed


# ── Desktop notification ──────────────────────────────────────────────────────

def _send_notification(title: str, body: str) -> None:
    """Send a best-effort desktop notification. Never raises."""
    import platform
    import subprocess
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(
                ["notify-send", "--urgency=normal", "--expire-time=8000", title, body],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            script = f'display notification "{body[:200]}" with title "{title}"'
            subprocess.Popen(["osascript", "-e", script],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Windows":
            ps = (f"Add-Type -AssemblyName System.Windows.Forms; "
                  f"[System.Windows.Forms.MessageBox]::Show('{body[:200]}','{title}')")
            subprocess.Popen(["powershell", "-Command", ps],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
