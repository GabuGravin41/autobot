"""
Claude Code Tool — thin Computer-submodule wrapper around
autobot/integrations/claude_code_bridge.py, mirroring how kaggle_tool.py
wraps the Kaggle API.

Why this thin wrapper exists (rather than reaching claude_code_bridge
directly from CoreLoop): Computer.get_tool_catalog() auto-discovers every
public method on every Computer submodule and advertises it to the LLM,
and autobot/computer/dispatch.py's AST-safe parser already lets the LLM
call any of them via the existing `computer_call` action
(`computer.claude_code.run(...)`). Wiring it in here means Autobot gets
Claude Code access with no new action name, no new CoreLoop dispatch
branch, and no new risk-classification code path — computer_call's
existing pattern-based classifier (see autobot/agent/approval.py) already
covers it, once it knows what to look for (see the DANGER pattern added
there for write-capable permission modes).
"""
from __future__ import annotations

from typing import Any


class ClaudeCode:
    """Computer.claude_code — headless Claude Code, not chat-window automation."""

    def run(
        self,
        prompt: str,
        cwd: str | None = None,
        permission_mode: str = "plan",
        allowed_tools: list | None = None,
        continue_session: bool = False,
        resume_session_id: str | None = None,
        timeout: float = 600.0,
    ) -> str:
        """
        Run one headless Claude Code turn and return its result text.

        permission_mode: "plan" (read-only, the safe default — never writes
        files or runs tools), "acceptEdits" (writes files without
        prompting), "bypassPermissions" (writes files AND runs tools
        without prompting — most capable, least safe). Avoid "default" —
        it prompts interactively, which headless mode can't answer.

        Raises RuntimeError with the bridge's error message on failure,
        so a bad call surfaces through computer_call's normal error path
        (dispatch.py catches Exception and returns it as agent-readable
        text) instead of silently returning nothing useful.
        """
        from autobot.integrations import claude_code_bridge

        result = claude_code_bridge.run_headless(
            prompt=prompt,
            cwd=cwd,
            permission_mode=permission_mode,
            allowed_tools=allowed_tools,
            continue_session=continue_session,
            resume_session_id=resume_session_id,
            timeout=timeout,
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "claude_code run failed"))

        data: Any = result.get("data")
        if isinstance(data, dict):
            # Prefer a human-readable "result" field if the CLI's JSON has
            # one; otherwise fall back to the whole parsed object as text.
            return str(data.get("result", data))
        return str(data)
