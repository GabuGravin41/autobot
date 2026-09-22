"""
Antigravity Tool — thin Computer-submodule wrapper around
autobot/integrations/antigravity_bridge.py, mirroring how kaggle_tool.py
wraps the Kaggle API and claude_code_tool.py wraps the Claude Code CLI.

Why this thin wrapper exists (rather than reaching antigravity_bridge
directly from CoreLoop): Computer.get_tool_catalog() auto-discovers every
public method on every Computer submodule and advertises it to the LLM,
and autobot/computer/dispatch.py's AST-safe parser already lets the LLM
call any of them via the existing `computer_call` action
(`computer.antigravity.run(...)`). Wiring it in here means Autobot gets
Antigravity access with no new action name, no new CoreLoop dispatch
branch, and no new risk-classification code path — computer_call's
existing pattern-based classifier (see autobot/agent/approval.py) already
covers it, once it knows what to look for (see the DANGER pattern added
there for skip_permissions=True, mirroring claude_code's write-capable
permission modes).

Deliberately CLI, not GUI automation: this session confirmed Antigravity
ships a real headless CLI (`agy`, see antigravity_bridge.py's docstring)
and separately confirmed — the hard way, via this project's own
computer-use tooling — that VS Code and browsers are only reachable at
click/read tier through screen automation, not typing. Reaching Antigravity
through its actual CLI sidesteps that limitation entirely rather than
working around it, and it's the same "API before UI automation" principle
DESIGN_PHILOSOPHY.md already commits this project to for every other
integration.
"""
from __future__ import annotations

from typing import Any


class Antigravity:
    """Computer.antigravity — headless Antigravity (agy CLI), not chat-window automation."""

    def run(
        self,
        prompt: str,
        cwd: str | None = None,
        skip_permissions: bool = False,
        model: str | None = None,
        effort: str | None = None,
        agent: str | None = None,
        continue_session: bool = False,
        conversation_id: str | None = None,
        timeout: float = 600.0,
    ) -> str:
        """
        Run one headless Antigravity turn and return its result text.

        skip_permissions: False (the safe default) respects whatever scoped
        allowlist is configured in ~/.gemini/antigravity-cli/settings.json
        on the target machine — headless mode can't answer an interactive
        permission prompt, so anything not pre-allowlisted just won't
        happen. True passes --dangerously-skip-permissions: agy approves
        all tool calls, including file writes and command execution,
        without asking. Avoid running without either an allowlist already
        configured or skip_permissions explicitly considered — an
        Antigravity conversation with neither will mostly just report back
        that it couldn't do anything, which looks like a bug but isn't one.

        conversation_id / continue_session resume a specific ongoing
        Antigravity conversation instead of starting fresh each call — this
        is what lets the orchestrator check in on "the same" project
        conversation over time rather than re-explaining context every run.

        Raises RuntimeError with the bridge's error message on failure, so
        a bad call surfaces through computer_call's normal error path
        (dispatch.py catches Exception and returns it as agent-readable
        text) instead of silently returning nothing useful.
        """
        from autobot.integrations import antigravity_bridge

        result = antigravity_bridge.run_headless(
            prompt=prompt,
            cwd=cwd,
            skip_permissions=skip_permissions,
            model=model,
            effort=effort,
            agent=agent,
            continue_session=continue_session,
            conversation_id=conversation_id,
            timeout=timeout,
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "antigravity run failed"))

        data: Any = result.get("data")
        if isinstance(data, dict):
            # Prefer the documented "response" field (agy's JSON envelope)
            # if present; claude_code_tool.py's equivalent looks for
            # "result" instead — the two CLIs' JSON envelopes use different
            # field names for the same concept, confirmed against each
            # CLI's own docs, not guessed. Fall back to the whole parsed
            # object as text if neither key is there.
            if "response" in data:
                return str(data.get("response", data))
            return str(data.get("result", data))
        return str(data)
