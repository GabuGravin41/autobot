"""
Computer Call Dispatch — the single, AST-safe way to turn an LLM-emitted
`computer.<module>.<method>(args)` string into a real method call.

Why this module exists
----------------------
`Computer.get_tool_catalog()` advertises every `computer.*` method to the LLM
in the system prompt. A now-deleted `prompts/system_prompt_full.md`
documented the `{"computer_call": {"call": "..."}}` action across hundreds
of lines (it also described a stale vision-first architecture superseded by
this project's actual CDP/DOM-index-first design, which is why it was
removed rather than merged back in). But `ActionModel` had no
`computer_call` field, so those calls were silently dropped by pydantic and
executed as "unknown action" — the agent could see the tools but never
invoke them. This module plus the `computer_call` action in agent/models.py
closes that gap.

`agent/background_runner.py` had its own private copy of this logic whose
docstring claimed it "uses the same AST-safe dispatch pattern as AgentLoop"
— a pattern AgentLoop did not actually have. Both now call into here, so the
parsing/security rules can't drift apart again.

Security model
--------------
The call string is NEVER `eval()`d. We parse it structurally:
  1. A regex extracts module/method/arg-text — only `\\w+` names are allowed,
     so no attribute traversal (`__class__`, `__globals__`) is expressible.
  2. Names starting with `_` are rejected outright — no private/dunder access.
  3. The module and method must exist as real public attributes of the live
     Computer instance (getattr, not import).
  4. Arguments are parsed with `ast.literal_eval`, which evaluates ONLY
     literals (str/int/float/bool/None/list/dict/tuple). A crafted argument
     like `__import__('os').system('...')` raises instead of executing.

This is defense in depth, not the only line of defense: risky calls are also
classified and gated by agent/approval.py before ever reaching this module.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# computer.<module>.<method>(<args>)  — e.g. computer.mouse.click(x=1, y=2)
_CALL_RE_3 = re.compile(r"^computer\.(\w+)\.(\w+)\((.*)\)$", re.DOTALL)
# computer.<callable>(<args>)         — e.g. computer.anti_sleep()
_CALL_RE_2 = re.compile(r"^computer\.(\w+)\((.*)\)$", re.DOTALL)

# Modules that touch the physical screen/input. Blocked in background mode,
# where the user may be actively using the machine for something else.
SCREEN_MODULES = frozenset({"mouse", "keyboard", "display", "window", "browser"})

# Result text longer than this is truncated before going back into the prompt,
# so a huge file read or command output can't blow up the context budget.
_MAX_RESULT_CHARS = 2000


class DispatchError(Exception):
    """Raised when a call string is malformed or refers to something unavailable."""


def parse_computer_call(call_str: str) -> tuple[str | None, str, list, dict]:
    """
    Parse a `computer.*` call string into (module_name, method_name, args, kwargs).

    module_name is None for top-level callables like `computer.anti_sleep()`.
    Raises DispatchError if the string is malformed or uses a private name.
    """
    call_str = (call_str or "").strip()
    if not call_str.startswith("computer."):
        raise DispatchError(f"call must start with 'computer.': {call_str!r}")

    match3 = _CALL_RE_3.match(call_str)
    match2 = None if match3 else _CALL_RE_2.match(call_str)

    if match3:
        module_name, method_name, args_str = match3.groups()
    elif match2:
        module_name, method_name, args_str = None, match2.group(1), match2.group(2)
    else:
        raise DispatchError(
            f"cannot parse call: {call_str!r} — expected computer.<module>.<method>(...)"
        )

    # No private/dunder access. The \w+ regex already blocks dotted traversal;
    # this blocks the remaining `_`-prefixed surface.
    for name in (module_name, method_name):
        if name and name.startswith("_"):
            raise DispatchError(f"access to private name '{name}' is not allowed")

    args: list = []
    kwargs: dict = {}
    if args_str.strip():
        try:
            tree = ast.parse(f"_f({args_str})", mode="eval")
        except SyntaxError as e:
            raise DispatchError(f"could not parse arguments: {e}") from e
        call_node = tree.body
        if not isinstance(call_node, ast.Call):
            raise DispatchError("arguments did not parse as a call")
        for node in call_node.args:
            try:
                args.append(ast.literal_eval(node))
            except (ValueError, SyntaxError) as e:
                raise DispatchError(
                    f"argument must be a literal (string/number/list/dict), got an expression: {e}"
                ) from e
        for kw in call_node.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError) as e:
                raise DispatchError(
                    f"keyword '{kw.arg}' must be a literal, got an expression: {e}"
                ) from e

    return module_name, method_name, args, kwargs


def resolve_target(computer: Any, module_name: str | None, method_name: str) -> Any:
    """Resolve the bound method on the live Computer instance, or raise DispatchError."""
    if module_name is None:
        target = getattr(computer, method_name, None)
        if target is None or not callable(target):
            raise DispatchError(f"unknown callable 'computer.{method_name}'")
        return target

    module = getattr(computer, module_name, None)
    if module is None:
        available = ", ".join(sorted(public_module_names(computer)))
        raise DispatchError(f"unknown module '{module_name}'. Available: {available}")

    method = getattr(module, method_name, None)
    if method is None or not callable(method):
        available = ", ".join(
            sorted(n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n, None)))
        )
        raise DispatchError(
            f"unknown method 'computer.{module_name}.{method_name}'. Available: {available}"
        )
    return method


def public_module_names(computer: Any) -> list[str]:
    """Public sub-module attribute names on a Computer instance (mouse, keyboard, ...)."""
    return [n for n in vars(computer) if not n.startswith("_")]


async def dispatch_computer_call(
    computer: Any,
    call_str: str,
    blocked_modules: frozenset[str] = frozenset(),
) -> tuple[bool, str]:
    """
    Execute a `computer.*` call string against a live Computer instance.

    Returns (success, result_text). Never raises — failures come back as
    (False, message) so the agent can read the error and self-correct on the
    next step rather than crashing the run.

    blocked_modules: module names to refuse (e.g. SCREEN_MODULES in background
    mode). Refusal is reported as a normal failure, not an exception.
    """
    try:
        module_name, method_name, args, kwargs = parse_computer_call(call_str)
    except DispatchError as e:
        return False, f"Error: {e}"

    if module_name and module_name in blocked_modules:
        return False, (
            f"[BLOCKED] '{module_name}' is not available in this mode "
            "(it touches the screen/input while running in the background). "
            "Use terminal, files, clipboard, research, kaggle, or vault instead."
        )

    try:
        method = resolve_target(computer, module_name, method_name)
    except DispatchError as e:
        return False, f"Error: {e}"

    try:
        if asyncio.iscoroutinefunction(method):
            result = await method(*args, **kwargs)
        else:
            # Computer methods are blocking (pyautogui, subprocess, CDP-over-sync).
            # Run off the event loop so the agent's async machinery keeps breathing.
            result = await asyncio.to_thread(method, *args, **kwargs)
    except Exception as e:
        logger.warning(f"computer_call failed: {call_str} → {e}")
        return False, f"Error executing {call_str}: {e}"

    return True, _format_result(call_str, result)


def _format_result(call_str: str, result: Any) -> str:
    """Render a return value for the agent's next prompt, capped in length."""
    if result is None:
        return f"OK: {call_str}"
    text = str(result)
    if len(text) > _MAX_RESULT_CHARS:
        text = text[:_MAX_RESULT_CHARS] + f"... [truncated, {len(str(result))} chars total]"
    return text
