"""
ApprovalGuard — Risk-based action gating for autonomous agent runs.

Three modes (set via AUTOBOT_APPROVAL_MODE env var or settings API):
  strict   — Ask user before ANY action in the CAUTION or DANGER tier
  balanced — Ask user only for DANGER-tier actions (default)
  trusted  — Never interrupt for SAFE/CAUTION/DANGER; run those automatically

Risk tiers:
  SAFE         — Navigation, clicks, typing, scrolling, screenshots, reading
  CAUTION      — Form submissions and other reversible, low-blast-radius actions
  DANGER       — Shell execution and other risky-but-recoverable actions
  IRREVERSIBLE — Deleting files/data, financial transactions, entering
                 credentials, sending messages or publishing under the
                 user's identity. ALWAYS requires a live "Allow" click —
                 no mode, including trusted, can auto-proceed. This mirrors
                 how the assistant building this system operates: certain
                 categories stay hard-gated regardless of how much trust
                 has been granted, because the cost of one bad action here
                 (deleted research data, an unwanted purchase, a leaked
                 credential) is not something "you can just undo."

Unattended mode (AUTOBOT_UNATTENDED=1, or ApprovalGuard(unattended=True)) —
a SEPARATE axis from `mode` above, not a fourth mode value. `mode` answers
"how much do I trust the agent's judgment"; `unattended` answers "is anyone
actually at the keyboard to click Allow right now." The user's own framing
for this (Sep 2026, deciding how the Kaggle-notebook-iteration workflow
should behave while they're away from the computer): "act like trusted
mode" for CAUTION/DANGER while unattended — REGARDLESS of what `mode` is
set to, since a `strict`-mode run left going overnight that just hangs on
the first CAUTION action accomplishes nothing — while the IRREVERSIBLE
floor stays exactly as hard as it always is: unattended does not mean "no
one will notice," it means "no one is here to say yes," and those are
different things for exactly the tier that can't be undone. Concretely:
  - CAUTION / DANGER, unattended=True: proceed automatically, logged and
    desktop-notified for later review, independent of `mode`.
  - IRREVERSIBLE, unattended=True: still never proceeds. The only change
    from the attended case is *how* it's blocked — no `wait_for_approval`
    call (which would sit on a `timeout`-second clock with nobody able to
    answer it), just an immediate, clearly logged skip, so the run moves on
    to its next action instead of stalling on a decision no one is present
    to make. AUTOBOT_AUTO_APPROVE=1 (a separate, pre-existing, deliberately
    blunt global override — see gate() below) still bypasses this if set;
    unattended mode does not imply it and does not set it.

Usage (from AgentLoop._execute_step):
    from autobot.agent.approval import ApprovalGuard, RiskTier
    guard = ApprovalGuard(mode="balanced")
    risk = guard.classify(action, element_context=element_context)
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
    SAFE         = "safe"
    CAUTION      = "caution"
    DANGER       = "danger"
    IRREVERSIBLE = "irreversible"


# ── Keyword sets for risk classification ──────────────────────────────────────
# IRREVERSIBLE is checked first and always wins — see gate()'s hard-block below.

_IRREVERSIBLE_PATTERNS = [
    # File / data deletion — both Unix and Windows/PowerShell forms, since
    # this project runs on Windows and the Unix-only patterns below would
    # miss `del /f /s /q`, `rd /s /q`, and Remove-Item entirely.
    r"\brm\s+-\w*r\w*\b", r"\brmdir\b", r"shutil\.rmtree", r"os\.remove", r"os\.unlink",
    r"\bdel\s+/[a-z]*f[a-z]*\b", r"\brd\s+/s\b", r"\brmdir\s+/s\b",
    r"remove-item\b.{0,40}(-recurse|-force)",
    # Database / disk destruction
    r"\bdrop\b.*\btable\b", r"\btruncate\b", r"\bdelete from\b",
    r"\bmkfs\b", r"\bdd\b.*\bof=\b", r"\bformat\s+[a-z]:\b",
    # Account-level deletion
    r"delete.?account", r"close.?account",
    # Financial transactions
    r"\bpurchase\b", r"\bcheckout\b", r"\bbuy now\b", r"\bplace order\b",
    r"credit.?card", r"stripe", r"paypal", r"\bwire transfer\b", r"\bcrypto.{0,10}(send|transfer|swap)\b",
    # Credential / sensitive-data entry — matched against the target FIELD's
    # attributes (type="password", aria-label, placeholder) via element_context,
    # not the typed value itself, since typed values are arbitrary and
    # uninformative on their own.
    r'type="password"', r"\bcvv\b", r"\bcard.?number\b", r"\baccount.?number\b",
    r"\brouting.?number\b", r"\bssn\b", r"\bsocial security\b", r"\bpassport\b",
    r"\bapi.?key\b", r"\bsecret.?key\b", r"\bprivate.?key\b",
    # Sending messages / publishing under the user's identity — irreversible
    # once it leaves the machine, regardless of how "safe" the content is.
    r"\bsend.{0,20}(email|message|mail|dm|notification)\b",
    r"\bemail.{0,20}send\b",
    r"\bslack\b.*\bsend\b|\bwhatsapp\b|\btelegram\b.*\bsend\b",
    r"\btweet\b",
    r"\bgit push\b", r"\bnpm publish\b",
    # A real Kaggle competition submission — consumes a limited daily
    # attempt and posts to a live leaderboard under the user's account.
    # Matched against the computer_call string itself (e.g.
    # `computer.kaggle.submit("comp", "file.csv", "msg")`), which is the
    # only way this method is reachable — see kaggle_tool.py's Kaggle.submit().
    # Deliberately separate from kernels_push/push_kernel, which just
    # re-runs a kernel in your own account and is SAFE-tier below — the
    # user drew this line explicitly: iterate on notebooks freely, but
    # submit always needs a live human decision, no matter how much trust
    # has been granted elsewhere.
    r"kaggle\.submit\(", r"competition_submit\b", r"competitions_submit\b",
    # submit_code_competition() (added Sep 2026 Round 7) is the Code-
    # Competition equivalent of submit() — binds a kernel version to a
    # real leaderboard submission via the `kaggle competitions submit
    # -k ... -v ...` CLI form (see kaggle_tool.py's module docstring for
    # why this one shells out to the CLI instead of the Python API). Same
    # IRREVERSIBLE reasoning as submit() above: it's a real, quota-
    # consuming leaderboard attempt, just reached through a different
    # kaggle_tool.py method for competitions that reject a bare CSV
    # upload.
    r"kaggle\.submit_code_competition\(",
]

# computer_call strings that should be SAFE even though the generic
# computer_call classifier (see core_loop.py's _classify_risk()) defaults
# everything else to CAUTION. Kaggle kernel create/edit/read methods only
# touch a kernel in the user's own account — never a competition
# leaderboard, never a submission attempt (that's submit(), always
# IRREVERSIBLE via the pattern above) — so they should never pause for
# approval in ANY mode, including strict. This is what actually lets
# "create a notebook, edit it, run it, look at the output" happen freely
# while a real competition submission still always stops for a human.
_SAFE_COMPUTER_CALL_PATTERNS = [
    r"kaggle\.(pull_kernel|push_kernel|kernel_status|kernel_output)\(",
    # get_leaderboard() and list_top_kernels() (added Sep 2026 Round 7) are
    # both read-only — a public leaderboard snapshot and a public kernel
    # listing, neither touches the user's own kernels, submissions, or
    # account state, so they belong in the same frictionless tier as the
    # kernel read/iterate methods above.
    r"kaggle\.(get_leaderboard|list_top_kernels)\(",
]
_SAFE_COMPUTER_CALL_RE = re.compile("|".join(_SAFE_COMPUTER_CALL_PATTERNS), re.IGNORECASE)

_DANGER_PATTERNS = [
    # Shell execution — risky (can do almost anything) but not itself
    # irreversible; individual destructive sub-commands are still caught
    # by _IRREVERSIBLE_PATTERNS above regardless of which action carries them.
    r"subprocess", r"os\.system", r"shell=True",
    # Claude Code headless runs (computer.claude_code.run(...)) requesting a
    # write-capable permission_mode — it can edit or create real files (and,
    # for bypassPermissions, run shell/tool calls) without prompting, which
    # is a materially different risk than the "plan" (read-only) default.
    # Matched against the computer_call string's permission_mode argument.
    r"accept.?edits", r"bypass.?permissions",
    # Antigravity headless runs (computer.antigravity.run(...)) requesting
    # skip_permissions=True — the same shape of risk as claude_code's
    # write-capable modes above (agy approves all tool calls, including
    # file writes and shell execution, without asking) via a materially
    # different flag name (--dangerously-skip-permissions), so it needs its
    # own pattern rather than reusing "bypass.?permissions". The default
    # (skip_permissions=False, respecting agy's own scoped allowlist) is
    # NOT matched here and falls through to computer_call's generic CAUTION
    # default — the same treatment claude_code.run()'s "plan" mode default
    # gets, deliberately kept consistent between the two integrations.
    r"skip_permissions\s*=\s*true", r"dangerously.?skip.?permissions",
]

_CAUTION_PATTERNS = [
    # Web form submission via browser (not method calls in Python code)
    r"\bform\.submit\b|\bsubmit form\b",
    r"\bgit commit\b",
]

_IRREVERSIBLE_RE = re.compile("|".join(_IRREVERSIBLE_PATTERNS), re.IGNORECASE)
_DANGER_RE = re.compile("|".join(_DANGER_PATTERNS), re.IGNORECASE)
_CAUTION_RE = re.compile("|".join(_CAUTION_PATTERNS), re.IGNORECASE)


def _action_text(action: "ActionModel", element_context: str = "") -> str:
    """Extract the human-readable text of an action for risk classification.

    For keyboard.type() and clipboard.set() calls, we strip out the typed
    content before classification — the danger patterns should evaluate what
    the agent is *doing*, not the arbitrary text it was asked to type.
    For terminal.run() and other shell-level calls, the full text is kept.

    Uses getattr() defensively: some fields referenced here (computer_call,
    input_text_native) are part of a planned generic tool-call action that
    isn't in the current ActionModel schema yet — accessing them directly
    would raise AttributeError and make the guard fail-crash instead of
    fail-safe. Once that action type exists, no change is needed here.
    """
    parts: list[str] = []
    computer_call = getattr(action, "computer_call", None)
    if computer_call:
        call = computer_call.call
        # Strip content of typing/clipboard actions — content is user-directed text,
        # not an agent action, so scanning it for "format", "rm", etc. causes false positives.
        if re.match(r"computer\.(keyboard\.type|clipboard\.set)\(", call):
            # Keep only the method name for classification
            parts.append(call.split("(")[0])
        else:
            parts.append(call)
    if action.navigate:
        parts.append(f"navigate to {action.navigate.url}")
    if action.input_text:
        parts.append("type_text")  # content excluded — see docstring
    input_text_native = getattr(action, "input_text_native", None)
    if input_text_native:
        parts.append("type_text_native")  # content excluded — see docstring
    if action.run_command:
        # Full command text IS kept — this is the actual mechanism by which
        # an agent would run `rm -rf`, `del /f /s /q`, `shutil.rmtree`, etc.
        parts.append(f"run_command: {action.run_command.command}")
    if action.done:
        parts.append(f"done: {action.done.text}")
    if element_context:
        parts.append(element_context)
    return " | ".join(parts) or action.action_name


# ── ApprovalGuard ─────────────────────────────────────────────────────────────

class ApprovalGuard:
    """
    Classifies each agent action by risk and gates it against the
    user's chosen approval mode.

    Instantiated once per AgentLoop run; mode is read from env at creation time
    so it can be changed between runs via the settings API.
    """

    def __init__(self, mode: str | None = None, unattended: bool | None = None) -> None:
        self.mode = (mode or os.getenv("AUTOBOT_APPROVAL_MODE", "balanced")).lower()
        if unattended is None:
            unattended = os.getenv("AUTOBOT_UNATTENDED", "0").strip().lower() in ("1", "true", "yes")
        self.unattended = unattended
        logger.info(
            f"ApprovalGuard active — mode: {self.mode}"
            + (" — UNATTENDED (CAUTION/DANGER auto-proceed, IRREVERSIBLE auto-skips)" if self.unattended else "")
        )

    def classify(self, action: "ActionModel", element_context: str = "") -> RiskTier:
        """Classify an action into SAFE / CAUTION / DANGER / IRREVERSIBLE.

        element_context: optional attributes of the DOM/native element being
        acted on (type, name, aria-label, placeholder) — pass this for
        input_text/input_text_native actions so credential-field detection
        (type="password", "card number", etc.) works off the FIELD, not the
        arbitrary text being typed into it.
        """
        text = _action_text(action, element_context=element_context)
        if _IRREVERSIBLE_RE.search(text):
            return RiskTier.IRREVERSIBLE
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
        timeout: float = 300.0,
        element_context: str = "",
    ) -> bool:
        """
        Gate an action based on current approval mode and risk tier.

        IRREVERSIBLE — ALWAYS pauses for explicit "Allow", in every mode
                        including trusted. No setting can bypass this tier;
                        that is the point of it. When unattended=True there
                        is no one to answer the pause, so it becomes an
                        immediate, logged skip instead of a wait — see the
                        module docstring's "Unattended mode" section.
        trusted       — DANGER proceeds automatically (with a notification);
                        CAUTION proceeds silently.
        balanced      — Pauses for DANGER; CAUTION proceeds with a log line.
        strict        — Pauses for CAUTION and DANGER too.
        unattended    — Independent of the above: CAUTION/DANGER proceed
                        exactly like trusted mode regardless of self.mode,
                        because self.mode's whole point is "how much should
                        I interrupt the person watching" and there isn't
                        one. Checked before the mode branches below, so it
                        always wins for these two tiers.

        Returns True to proceed, False to skip this action.
        """
        text = _action_text(action, element_context=element_context)
        tier_label = tier.value.upper()

        if tier == RiskTier.SAFE:
            return True

        if tier == RiskTier.IRREVERSIBLE:
            if os.getenv("AUTOBOT_AUTO_APPROVE") == "1":
                logger.warning(f"[AUTO_APPROVE/IRREVERSIBLE] Proceeding automatically: {text[:100]}")
                return True
            if self.unattended:
                logger.warning(
                    f"⛔ [UNATTENDED/IRREVERSIBLE] Skipped — no one present to approve: {text[:150]}"
                )
                _send_notification(
                    title="Autobot skipped an IRREVERSIBLE action while unattended",
                    body=f"{text[:120]}\nThis needed a live Allow and no one was there, so it did not run. "
                         f"Review the log when you're back.",
                )
                return False
            return await self._request_approval(text, tier_label, goal, timeout)

        if self.unattended and tier in (RiskTier.CAUTION, RiskTier.DANGER):
            logger.warning(f"[UNATTENDED/{tier_label}] Proceeding without pause: {text[:100]}")
            _send_notification(
                title=f"Autobot — {tier_label} action (unattended)",
                body=f"Doing: {text[:120]}\nYou were away, so this proceeded automatically. Review the log.",
            )
            return True

        if self.mode == "trusted":
            if tier == RiskTier.DANGER:
                logger.warning(f"[TRUSTED/DANGER] Proceeding without pause: {text[:100]}")
                _send_notification(
                    title="Autobot — Risky Action (trusted mode)",
                    body=f"Doing: {text[:120]}\nYou can pause or abort from the Autobot dashboard.",
                )
            return True  # trusted never pauses for DANGER or below

        if self.mode == "balanced" and tier == RiskTier.CAUTION:
            logger.info(f"[BALANCED/CAUTION] Proceeding: {text[:100]}")
            return True  # balanced only pauses for DANGER and above

        # strict: pause for CAUTION + DANGER too
        return await self._request_approval(text, tier_label, goal, timeout)

    async def _request_approval(self, text: str, tier_label: str, goal: str, timeout: float) -> bool:
        """Pause and wait for an explicit human Allow/Block via the dashboard."""
        from autobot.agent.human_gate import wait_for_approval

        key = "approval_" + hashlib.md5(text.encode()).hexdigest()[:10]
        message = (
            f"[{tier_label}] Agent wants to:\n{text[:200]}"
            + (f"\n\nCurrent goal: {goal[:200]}" if goal else "")
            + (
                "\n\nThis action cannot be undone — review carefully before allowing."
                if tier_label == "IRREVERSIBLE"
                else f"\n\nMode: {self.mode} — click Allow to proceed or Block to skip."
            )
        )
        logger.warning(f"⛔ Approval required ({tier_label}): {text[:100]}")
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
