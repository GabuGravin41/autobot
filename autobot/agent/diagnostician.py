"""
CLI Stderr Diagnostician — Parses shell command execution failures and tracebacks into structured repair prompts.

Component 43 of the Autobot 2.0 Master Roadmap.
Categorizes errors (Missing Module, Syntax Error, Timeout, Permission Error, File Not Found)
and structures payloads to be sent to web-based AIs (Grok, ChatGPT, Claude) for instant fix generation.
"""
import re
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

ErrorCategory = Literal[
    "missing_module",
    "syntax_error",
    "file_not_found",
    "permission_denied",
    "timeout",
    "unknown_error",
]


@dataclass
class DiagnosticResult:
    category: ErrorCategory
    summary: str
    suggested_action: str
    suggested_fix_prompt: str


class TerminalStderrDiagnostician:
    """
    Parses CLI stderr tracebacks and generates targeted prompt payloads for LLM auto-repair.
    """

    @staticmethod
    def analyze_stderr(command: str, stderr: str, exit_code: int = 1) -> DiagnosticResult:
        """
        Analyze command output stderr and categorize failure mode.
        """
        stderr_clean = stderr.strip()

        # 1. Missing Module / ImportError
        match_import = re.search(r"ModuleNotFoundError: No module named '([^']+)'", stderr_clean)
        if match_import:
            mod_name = match_import.group(1)
            return DiagnosticResult(
                category="missing_module",
                summary=f"Missing Python module '{mod_name}'.",
                suggested_action=f"Run `pip install {mod_name}` via `run_command`.",
                suggested_fix_prompt=f"The command '{command}' failed because module '{mod_name}' is missing. Install it using pip.",
            )

        # 2. SyntaxError
        if "SyntaxError:" in stderr_clean or "IndentationError:" in stderr_clean:
            return DiagnosticResult(
                category="syntax_error",
                summary="Python SyntaxError or IndentationError detected.",
                suggested_action="Pass traceback to Grok/ChatGPT tab to fix script syntax.",
                suggested_fix_prompt=f"The script executed via '{command}' had a SyntaxError:\n{stderr_clean[:500]}\nRewrite the script correctly.",
            )

        # 3. FileNotFoundError
        match_fnf = re.search(r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'", stderr_clean)
        if match_fnf:
            missing_file = match_fnf.group(1)
            return DiagnosticResult(
                category="file_not_found",
                summary=f"File not found: '{missing_file}'.",
                suggested_action=f"Verify or create file '{missing_file}' before retrying.",
                suggested_fix_prompt=f"Command '{command}' failed because file '{missing_file}' does not exist.",
            )

        # 4. Permission Error
        if "PermissionError" in stderr_clean or "Access is denied" in stderr_clean:
            return DiagnosticResult(
                category="permission_denied",
                summary="Permission error or file access locked.",
                suggested_action="Use alternative output path or request human permission.",
                suggested_fix_prompt=f"Command '{command}' hit permission denied: {stderr_clean[:300]}",
            )

        # Generic Fallback
        return DiagnosticResult(
            category="unknown_error",
            summary=f"Command failed with exit code {exit_code}.",
            suggested_action="Inspect stderr traceback and adjust strategy.",
            suggested_fix_prompt=f"Command '{command}' failed with error:\n{stderr_clean[:500]}",
        )
