"""
Step Verifier -- Dense, immediate credit assignment for every agent step.

Tesla V12 lesson applied:
  Tesla proxy tasks fire an immediate gradient even when the primary action
  (steering) has not changed yet.  A distant pedestrian may not require
  braking for several seconds, but the segmentation head error fires NOW.

  StepVerifier is the equivalent auxiliary head.  After every action batch,
  it compares the LLM predicted state delta (proxy_assessment
  .expected_state_delta) against what actually happened (URL, action results).
  The verdict is stored on StepHistoryEntry.verification_result and injected
  into the next step history -- giving the model immediate feedback before
  downstream consequences of a wrong assumption appear in the control output.

Distinction from JudgeAgent (runs once at end):
  StepVerifier is synchronous, rule-based, runs after EVERY step.
  No LLM call.  It verifies the specific prediction the model made THIS step.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autobot.agent.models import ActionResult, ProxyAssessment

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """The verifier verdict on one step execution."""
    passed: bool
    label: str   # VERIFIED | PARTIAL | FAILED | SILENT_FAIL | SKIPPED
    reason: str
    predicted: str = ""

    @property
    def history_text(self) -> str:
        """Format for injection into agent step history."""
        if self.passed:
            icon = "[OK]"
        elif self.label == "PARTIAL":
            icon = "[WARN]"
        else:
            icon = "[FAIL]"
        parts = [f"{icon}[{self.label}] {self.reason}"]
        if self.predicted:
            preview = self.predicted[:120]
            parts.append(f'    Predicted: "{preview}"')
        return "\n  ".join(parts)


class StepVerifier:
    """
    Lightweight, rule-based post-step verifier.

    Runs synchronously after every _execute_actions() in AgentLoop._execute_step().
    No LLM call, no async I/O.

    Strategy (priority order):
    1. Any action failed             -> FAILED
    2. Prediction mentions URL/nav   -> check url_before vs url_after
    3. Prediction mentions keywords  -> check extracted_content
    4. All actions succeeded         -> VERIFIED (soft)
    """

    _URL_HINTS = re.compile(
        r"url.{0,10}change|navigate|redirect|go to|page.{0,10}load|page.{0,10}open",
        re.IGNORECASE,
    )
    _RELOAD_HINTS = re.compile(r"reload|refresh|submit", re.IGNORECASE)

    def verify(
        self,
        proxy_assessment: "ProxyAssessment | None",
        action_results: "list[ActionResult]",
        url_before: str,
        url_after: str,
    ) -> VerificationResult:
        """Run post-step verification. Returns a VerificationResult."""
        if proxy_assessment is None or not proxy_assessment.expected_state_delta:
            return VerificationResult(
                passed=True,
                label="SKIPPED",
                reason="No proxy_assessment provided -- skipping verification.",
            )

        predicted = proxy_assessment.expected_state_delta.strip()

        # 1. Hard failure: any action explicitly failed
        failed = [r for r in action_results if not r.success]
        if failed:
            errs = "; ".join(
                r.action_name + ": " + (r.error or "no detail")[:80]
                for r in failed[:3]
            )
            return VerificationResult(
                passed=False,
                label="FAILED",
                reason="Action(s) failed -- predicted delta not achieved. Failures: " + errs,
                predicted=predicted,
            )

        # 2. URL/navigation prediction
        url_changed = url_before != url_after
        pred_nav = bool(self._URL_HINTS.search(predicted)) or bool(self._RELOAD_HINTS.search(predicted))
        tgt = self._extract_url(predicted)
        if pred_nav:
            if not url_changed:
                return VerificationResult(
                    passed=False,
                    label="SILENT_FAIL",
                    reason=(
                        "Predicted a URL/page change but URL stayed ("
                        + url_before[:60]
                        + "). Action may have silently failed (overlay, disabled element, cancelled redirect)."
                    ),
                    predicted=predicted,
                )
            if tgt and tgt not in url_after:
                return VerificationResult(
                    passed=False,
                    label="PARTIAL",
                    reason=(
                        "URL changed to " + url_after[:60]
                        + " but not to predicted target " + tgt
                        + ". Likely intermediate page (login wall, error, redirect)."
                    ),
                    predicted=predicted,
                )
            return VerificationResult(
                passed=True,
                label="VERIFIED",
                reason="URL changed as predicted -> " + url_after[:80],
                predicted=predicted,
            )

        # 3. Keyword match in extracted content
        content = " ".join(
            r.extracted_content for r in action_results if r.extracted_content
        )
        kws = self._keywords(predicted)
        if kws and content:
            hits = [k for k in kws if k.lower() in content.lower()]
            if hits:
                return VerificationResult(
                    passed=True,
                    label="VERIFIED",
                    reason="Predicted keywords found in action output: " + str(hits[:5]),
                    predicted=predicted,
                )

        # 4. Soft success
        return VerificationResult(
            passed=True,
            label="VERIFIED",
            reason="All actions succeeded. No contradiction detected (delta unmeasured).",
            predicted=predicted,
        )

    @staticmethod
    def _extract_url(text: str) -> str:
        m = re.search(r"https?://[^\s]+" + r"|/[a-zA-Z0-9_\-/]+", text)
        return m.group(0) if m else ""

    @staticmethod
    def _keywords(text: str) -> list:
        stop = {
            "will", "the", "a", "an", "to", "of", "in", "on", "at", "and",
            "or", "is", "it", "that", "this", "for", "with", "url", "page",
            "tab", "show", "change", "appear", "should", "expected",
        }
        return [w for w in re.findall(r"[a-zA-Z]{4,}", text) if w.lower() not in stop][:8]
