# Engine tool-call review

This document records that engine actions have been reviewed and hardened so that missing or invalid arguments do not crash the runner with `KeyError`, and clipboard failures are handled gracefully.

## Engine actions (autobot/engine.py `_execute_action`)

| Action | Defensive behavior |
|--------|---------------------|
| **open_url** | Uses `args.get("url", "")`; if empty, logs and returns message (no browser call). |
| **search_google** | Uses `args.get("query", "")`; if empty, logs and returns (no browser call). |
| **browser_fill** | `args.get("selector")` / `args.get("text")`; raises `ValueError` if selector missing. |
| **browser_click** | `args.get("selector")`; raises `ValueError` if selector missing. |
| **browser_click_text** | `args.get("text")`; raises `ValueError` if text missing. |
| **browser_press** | `args.get("key", "enter")`; defaults to `"enter"` if missing. |
| **browser_read_text** | `args.get("selector")`; raises `ValueError` if selector missing. |
| **open_app** | `args.get("command", "")`; if empty, logs and returns (no subprocess). |
| **open_path** | `args.get("path", "")`; raises `ValueError` if path empty. |
| **run_command** | `args.get("command", "")`; if empty, sets `last_command_output` to message and returns (no subprocess). |
| **desktop_click** | `args.get("x", 0)` / `args.get("y", 0)`; raises `ValueError` if conversion to int fails. |
| **desktop_move** | Same as desktop_click. |
| **clipboard_set** | Wrapped in try/except; on failure (e.g. PowerShell), logs and returns `""`. |
| **clipboard_get** | Wrapped in try/except; on failure, logs and returns `""`. |
| **write_file** | Existing check: raises `ValueError` if `path` missing. |
| **state_set** | Existing check: raises `ValueError` if `key` missing. |
| **request_human_input** | Uses `args.get("key", ...)` and `args.get("prompt", ...)`; safe. |
| **notify_user** | Uses `args.get("message", ...)`; safe. |
| **screenshot** | Validates `filename` and `run_dir`; raises clear `ValueError` if missing. |
| **adapter_run** / **adapter_confirm** | Use `args.get(...)` for adapter_name, adapter_action, params; params default to `{}`. |

Workflows use `continue_on_error=True` on steps that can fail (browser, adapter, clipboard, etc.), so a step that raises `ValueError` is caught and the run continues.
