"""
Offline tests for ApprovalGuard's unattended mode (autobot/agent/approval.py),
added Sep 2026 for the Kaggle kernel-iteration workflow running while the
user is away from the computer — see approval.py's module docstring
("Unattended mode") for the full rationale. Two things must both hold:

  1. CAUTION/DANGER proceed automatically while unattended, REGARDLESS of
     `mode` — including `strict`, which would otherwise pause for both.
  2. IRREVERSIBLE never proceeds just because no one's there to say yes —
     it's still blocked, but the block is immediate (no wait_for_approval
     call, no timeout clock ticking on a decision nobody can make), not a
     silent auto-approval and not a multi-minute hang either.

These go through the real ApprovalGuard.gate() + the real _ActionStub
CoreLoop uses (core_loop.py's _gate()), not a hand-rolled stub, so a
mismatch between the two can't hide from these tests the way it could if
this file re-implemented its own action shim.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

from autobot.agent.action_models import Action
from autobot.agent.approval import ApprovalGuard, RiskTier
from autobot.agent.core_loop import _ActionStub


def _run(coro):
    return asyncio.run(coro)


def _stub(name: str, params: dict | None = None) -> _ActionStub:
    return _ActionStub(Action(name=name, params=params or {}))


# A CAUTION-tier stub: computer_call not matching any DANGER/IRREVERSIBLE
# pattern falls through _classify_risk()'s computer_call branch to CAUTION
# (see core_loop.py) — using kaggle.list_competitions(), which
# test_approval_new_patterns.py already pins as NOT matching the SAFE
# kernel-ops pattern either, so it's genuinely CAUTION here.
def _caution_stub() -> _ActionStub:
    return _stub("computer_call", {"call": "computer.kaggle.list_competitions()"})


def _danger_stub() -> _ActionStub:
    return _stub("computer_call", {"call": 'computer.claude_code.run("x", permission_mode="acceptEdits")'})


def _irreversible_stub() -> _ActionStub:
    return _stub("computer_call", {"call": 'computer.kaggle.submit("comp", "sub.csv", "msg")'})


class TestUnattendedCautionAndDangerAutoProceed:
    def test_caution_proceeds_when_unattended_even_in_strict_mode(self):
        guard = ApprovalGuard(mode="strict", unattended=True)
        allowed = _run(guard.gate(_caution_stub(), RiskTier.CAUTION))
        assert allowed is True

    def test_danger_proceeds_when_unattended_even_in_strict_mode(self):
        guard = ApprovalGuard(mode="strict", unattended=True)
        allowed = _run(guard.gate(_danger_stub(), RiskTier.DANGER))
        assert allowed is True

    def test_unattended_does_not_call_request_approval_for_caution_danger(self):
        # If this ever regressed to calling _request_approval instead of
        # short-circuiting, it would try to await wait_for_approval and
        # either hang on its timeout or need a human — neither acceptable
        # while unattended. Assert the human-approval path is never touched.
        guard = ApprovalGuard(mode="strict", unattended=True)
        with patch("autobot.agent.human_gate.wait_for_approval") as mock_wait:
            _run(guard.gate(_caution_stub(), RiskTier.CAUTION))
            _run(guard.gate(_danger_stub(), RiskTier.DANGER))
        mock_wait.assert_not_called()


class TestUnattendedIrreversibleStillBlocks:
    def test_irreversible_returns_false_when_unattended(self):
        guard = ApprovalGuard(mode="trusted", unattended=True)
        allowed = _run(guard.gate(_irreversible_stub(), RiskTier.IRREVERSIBLE))
        assert allowed is False

    def test_irreversible_unattended_never_waits_for_human(self):
        # The whole point: no wait_for_approval call, no timeout clock —
        # an immediate, logged skip. Mock it and assert it's never reached.
        guard = ApprovalGuard(mode="trusted", unattended=True)
        with patch("autobot.agent.human_gate.wait_for_approval") as mock_wait:
            allowed = _run(guard.gate(_irreversible_stub(), RiskTier.IRREVERSIBLE, timeout=300.0))
        mock_wait.assert_not_called()
        assert allowed is False

    def test_auto_approve_env_still_overrides_even_when_unattended(self):
        # AUTOBOT_AUTO_APPROVE is a separate, pre-existing, deliberately
        # blunt override (documented in approval.py) — unattended mode must
        # not disable it, but it also must not be implied by it either
        # (see the next test).
        guard = ApprovalGuard(mode="trusted", unattended=True)
        with patch.dict(os.environ, {"AUTOBOT_AUTO_APPROVE": "1"}):
            allowed = _run(guard.gate(_irreversible_stub(), RiskTier.IRREVERSIBLE))
        assert allowed is True

    def test_unattended_alone_does_not_set_auto_approve(self):
        # Sanity check for the inverse of the above: unattended=True with
        # AUTOBOT_AUTO_APPROVE unset/0 must still block IRREVERSIBLE.
        guard = ApprovalGuard(mode="trusted", unattended=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOBOT_AUTO_APPROVE", None)
            allowed = _run(guard.gate(_irreversible_stub(), RiskTier.IRREVERSIBLE))
        assert allowed is False


class TestAttendedBehaviorUnchanged:
    """Guards against the unattended-mode change accidentally altering the
    existing, already-tested attended-mode behavior (regression check)."""

    def test_strict_still_pauses_for_caution_when_attended(self):
        guard = ApprovalGuard(mode="strict", unattended=False)
        with patch("autobot.agent.human_gate.wait_for_approval", AsyncMock(return_value=True)) as mock_wait:
            allowed = _run(guard.gate(_caution_stub(), RiskTier.CAUTION))
        mock_wait.assert_called_once()
        assert allowed is True  # approved via the (mocked) human path

    def test_balanced_still_auto_proceeds_caution_when_attended(self):
        guard = ApprovalGuard(mode="balanced", unattended=False)
        with patch("autobot.agent.human_gate.wait_for_approval") as mock_wait:
            allowed = _run(guard.gate(_caution_stub(), RiskTier.CAUTION))
        mock_wait.assert_not_called()  # balanced doesn't pause for CAUTION at all
        assert allowed is True

    def test_irreversible_still_waits_for_human_when_attended(self):
        guard = ApprovalGuard(mode="trusted", unattended=False)
        with patch("autobot.agent.human_gate.wait_for_approval", AsyncMock(return_value=False)) as mock_wait:
            allowed = _run(guard.gate(_irreversible_stub(), RiskTier.IRREVERSIBLE))
        mock_wait.assert_called_once()
        assert allowed is False


class TestUnattendedFromEnvVar:
    def test_env_var_true_sets_unattended(self):
        with patch.dict(os.environ, {"AUTOBOT_UNATTENDED": "1"}):
            guard = ApprovalGuard(mode="balanced")
        assert guard.unattended is True

    def test_env_var_absent_defaults_to_attended(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOBOT_UNATTENDED", None)
            guard = ApprovalGuard(mode="balanced")
        assert guard.unattended is False

    def test_explicit_kwarg_overrides_env_var(self):
        with patch.dict(os.environ, {"AUTOBOT_UNATTENDED": "1"}):
            guard = ApprovalGuard(mode="balanced", unattended=False)
        assert guard.unattended is False
