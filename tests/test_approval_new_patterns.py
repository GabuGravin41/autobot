"""
Offline tests for the risk-classification patterns added to
autobot/agent/approval.py alongside the Kaggle/Claude Code integrations:

  1. IRREVERSIBLE: a real Kaggle competition submission
     (`computer.kaggle.submit(...)`) — consumes a limited daily attempt and
     posts to a live leaderboard. Must hard-gate in every approval mode,
     same as file deletion or a financial transaction.
  2. DANGER: a Claude Code headless run requesting a write-capable
     permission_mode (`acceptEdits` / `bypassPermissions`) — can edit or
     create real files without prompting.
  3. SAFE: Kaggle kernel create/edit/read (`pull_kernel`/`push_kernel`/
     `kernel_status`/`kernel_output`) — the user drew this line explicitly:
     iterate on a notebook freely with zero friction, in every approval
     mode including strict, but a real competition submission always stops
     for a live human decision. Without this pattern, these calls would
     fall through to computer_call's CAUTION default, which is harmless in
     balanced/trusted mode but would still pause in strict mode — this
     pattern makes iteration frictionless unconditionally.

These test the exact regexes CoreLoop._classify_risk() uses for the
`computer_call` action (via approval.py's _IRREVERSIBLE_RE / _DANGER_RE /
_SAFE_COMPUTER_CALL_RE), since that's the real code path — not just
approval.py's higher-level ApprovalGuard.classify(), which operates on a
different (ActionModel-based) input shape CoreLoop no longer uses.
"""
from __future__ import annotations

from autobot.agent.approval import _DANGER_RE, _IRREVERSIBLE_RE, _SAFE_COMPUTER_CALL_RE


class TestKaggleSubmitIsIrreversible:
    def test_kaggle_submit_call_matches(self):
        call = 'computer.kaggle.submit("titanic", "submission.csv", "attempt 3")'
        assert _IRREVERSIBLE_RE.search(call) is not None

    def test_kaggle_pull_kernel_does_not_match(self):
        call = 'computer.kaggle.pull_kernel("user/my-kernel", "./work")'
        assert _IRREVERSIBLE_RE.search(call) is None

    def test_kaggle_push_kernel_does_not_match_irreversible(self):
        # push_kernel re-runs a kernel in the user's own account — SAFE
        # (see TestKaggleKernelOpsAreSafe below), never IRREVERSIBLE.
        call = 'computer.kaggle.push_kernel("./work")'
        assert _IRREVERSIBLE_RE.search(call) is None

    def test_kaggle_status_and_output_do_not_match(self):
        assert _IRREVERSIBLE_RE.search('computer.kaggle.kernel_status("user/k")') is None
        assert _IRREVERSIBLE_RE.search('computer.kaggle.kernel_output("user/k", "./out")') is None


class TestClaudeCodeWriteModeIsDanger:
    def test_accept_edits_matches(self):
        call = 'computer.claude_code.run("edit foo.py", permission_mode="acceptEdits")'
        assert _DANGER_RE.search(call) is not None

    def test_bypass_permissions_matches(self):
        call = 'computer.claude_code.run("do anything", permission_mode="bypassPermissions")'
        assert _DANGER_RE.search(call) is not None

    def test_plan_mode_does_not_match(self):
        call = 'computer.claude_code.run("read this code", permission_mode="plan")'
        assert _DANGER_RE.search(call) is None

    def test_case_insensitive(self):
        assert _DANGER_RE.search('permission_mode="ACCEPTEDITS"') is not None
        assert _DANGER_RE.search('permission_mode="bypasspermissions"') is not None


class TestAntigravitySkipPermissionsIsDanger:
    """Same shape of risk as TestClaudeCodeWriteModeIsDanger above, added
    Sep 2026 alongside the Antigravity CLI integration — a materially
    different flag name (--dangerously-skip-permissions vs
    acceptEdits/bypassPermissions) needed its own pattern rather than
    accidentally relying on the Claude Code one matching by coincidence."""

    def test_skip_permissions_true_matches(self):
        call = 'computer.antigravity.run("edit foo.py", skip_permissions=True)'
        assert _DANGER_RE.search(call) is not None

    def test_skip_permissions_false_does_not_match(self):
        call = 'computer.antigravity.run("read this code", skip_permissions=False)'
        assert _DANGER_RE.search(call) is None

    def test_default_call_with_no_skip_permissions_arg_does_not_match(self):
        call = 'computer.antigravity.run("what is the status of this project")'
        assert _DANGER_RE.search(call) is None

    def test_case_insensitive(self):
        assert _DANGER_RE.search('skip_permissions=TRUE') is not None
        assert _DANGER_RE.search('SKIP_PERMISSIONS=true') is not None

    def test_default_call_is_not_irreversible_or_safe(self):
        # Falls through to computer_call's generic CAUTION default — same
        # treatment claude_code.run()'s "plan" mode default gets, not an
        # explicit SAFE pattern (unlike Kaggle's kernel-ops carve-out).
        call = 'computer.antigravity.run("what is the status of this project")'
        assert _IRREVERSIBLE_RE.search(call) is None
        assert _SAFE_COMPUTER_CALL_RE.search(call) is None


class TestKaggleKernelOpsAreSafe:
    """pull/push/status/output — create, edit, run, and read a notebook in
    the user's own account. Explicitly SAFE so this never pauses for
    approval, in any mode (including strict), while submit() (tested above)
    always does. This is the exact separation the user asked for."""

    def test_push_kernel_is_safe(self):
        assert _SAFE_COMPUTER_CALL_RE.search('computer.kaggle.push_kernel("./work")') is not None

    def test_pull_kernel_is_safe(self):
        assert _SAFE_COMPUTER_CALL_RE.search('computer.kaggle.pull_kernel("user/k", "./work")') is not None

    def test_kernel_status_is_safe(self):
        assert _SAFE_COMPUTER_CALL_RE.search('computer.kaggle.kernel_status("user/k")') is not None

    def test_kernel_output_is_safe(self):
        assert _SAFE_COMPUTER_CALL_RE.search('computer.kaggle.kernel_output("user/k", "./out")') is not None

    def test_submit_does_not_match_safe_pattern(self):
        # submit() must never accidentally qualify as SAFE — IRREVERSIBLE
        # is checked first in _classify_risk() regardless, but this pins
        # down that the SAFE pattern itself doesn't overlap with it.
        call = 'computer.kaggle.submit("comp", "sub.csv", "msg")'
        assert _SAFE_COMPUTER_CALL_RE.search(call) is None

    def test_unrelated_kaggle_methods_do_not_match(self):
        # list_competitions/get_leaderboard/download_data were not part of
        # what the user asked to loosen — they stay at computer_call's
        # CAUTION default, not SAFE.
        assert _SAFE_COMPUTER_CALL_RE.search('computer.kaggle.list_competitions()') is None
        assert _SAFE_COMPUTER_CALL_RE.search('computer.kaggle.get_leaderboard("comp")') is None
        assert _SAFE_COMPUTER_CALL_RE.search('computer.kaggle.download_data("comp", "./data")') is None
