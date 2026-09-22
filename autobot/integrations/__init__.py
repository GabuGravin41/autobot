"""
autobot.integrations — API-level bridges to other tools, deliberately NOT
browser/UI automation of them.

This package exists because of a specific lesson from this project's own
history (see ROADMAP.md / DESIGN_PHILOSOPHY.md): automating another tool's
UI is fragile and expensive in tokens; calling its actual API or CLI, when
one exists, is an order of magnitude more reliable. autobot/browser/
extension_bridge.py is the exception that proves the rule — it exists
because Overleaf has no public write API, so DOM automation is the only
option there. Kaggle and Claude Code both ship real, documented,
non-interactive interfaces, so those go through subprocess calls to their
official CLIs instead:

  kaggle_bridge.py      — wraps the `kaggle` CLI (kernels pull/push/status/output)
  claude_code_bridge.py — wraps `claude -p` (Claude Code's headless mode)

Both modules follow the same contract as autobot/browser/extension_bridge.py's
ExtensionBridge.run(): every public function returns
    {"ok": bool, "data": <str|dict|None>, "error": str}
and never raises, so CoreLoop's dispatcher can call them uniformly.
"""
