"""
Tolerant JSON extraction and action-shape coercion for weaker models.

The gemma4 competition log (competitions/gemma_4_developer_agent/
THINKING_AND_DECISIONS.md §7) recorded exactly how a smaller model fails a
prompted JSON protocol: flat args instead of nested ones, a list where a
dict was expected, bare `name(args)` pseudo-calls instead of JSON, and no
convergence even after being told it was wrong. The fix that worked there
was making the PARSER permissive rather than hoping the model gets precise.

Two entry points:

    extract_json(text)                  -> the first JSON-ish object in text, or None
    coerce_call(data, known_params)     -> (name, params) or None

`coerce_call` never guesses a name that isn't in `known_params`. An
unrecognized shape returns None so the caller can re-ask — it must never be
turned into a default action (the old parser turned any unrecognized shape
into a successful `done`).
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON|python|tool_call)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _balanced_objects(text: str) -> list[str]:
    """Every top-level {...} span in text, string-literal aware."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_str: str | None = None
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
                start = -1
    return spans


def _try_parse(candidate: str) -> Any:
    candidate = candidate.strip()
    if not candidate:
        return None
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            pass
    try:  # single quotes, True/False/None — python-literal dicts
        value = ast.literal_eval(candidate)
        if isinstance(value, (dict, list)):
            return value
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    return None


def extract_json(text: str | None) -> Any:
    """Return the first parseable JSON object (dict) found in `text`, else None.

    Tries, in order: the whole text, fenced blocks, then every balanced
    {...} span. Prefers dicts; a bare list is returned only if nothing else
    parses.
    """
    if not text:
        return None
    candidates = [text]
    candidates += _FENCE_RE.findall(text)
    candidates += _balanced_objects(text)
    fallback = None
    for c in candidates:
        value = _try_parse(c)
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and fallback is None:
            fallback = value
    return fallback


_NAME_KEYS = ("name", "action", "type", "tool", "tool_name", "function", "move", "command_name")
_PARAM_KEYS = ("params", "parameters", "args", "arguments", "input", "kwargs")
_PSEUDO_CALL_RE = re.compile(r"^\s*(?:call:\s*)?([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", re.DOTALL)


def _as_params(value: Any, name: str, known_params: dict[str, list[str]]) -> dict[str, Any] | None:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = _try_parse(value)
        if isinstance(parsed, dict):
            return parsed
        order = known_params.get(name) or []
        return {order[0]: value} if order else None
    if isinstance(value, (list, tuple)):
        order = known_params.get(name) or []
        if len(value) > len(order):
            return None
        return {order[i]: v for i, v in enumerate(value)}
    order = known_params.get(name) or []
    return {order[0]: value} if order else None


def _parse_pseudo_call(text: str, known_params: dict[str, list[str]]) -> tuple[str, dict] | None:
    m = _PSEUDO_CALL_RE.match(text.strip())
    if not m:
        return None
    name = m.group(1).split(".")[-1]
    if name not in known_params:
        return None
    try:
        call = ast.parse(f"f({m.group(2)})", mode="eval").body
        assert isinstance(call, ast.Call)
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg}
    except Exception:
        return None
    order = known_params.get(name) or []
    if len(args) > len(order):
        return None
    params = {order[i]: v for i, v in enumerate(args)}
    params.update(kwargs)
    return name, params


def coerce_call(data: Any, known_params: dict[str, list[str]]) -> tuple[str, dict[str, Any]] | None:
    """
    Map the many shapes a model might emit onto (name, params).

    known_params: {action_name: [positional parameter order]} — used both to
    validate the name and to map positional/list/scalar args onto names.

    Recognized shapes (all observed from real models):
      {"name": "click", "params": {"index": 3}}             canonical
      {"name": "click", "args": {...}} / "arguments"/"input"
      {"type": "click", "index": 3}                         name key + flat params
      {"tool": "run_shell", "args": ["dir"]}                positional list
      {"click": {"index": 3}}                               single key = action name
      {"function": {"name": "click", "arguments": "{...}"}} OpenAI tool-call shape
      "click(3)" / "call: click(index=3)"                   pseudo-python
    """
    if isinstance(data, str):
        return _parse_pseudo_call(data, known_params)
    if not isinstance(data, dict) or not data:
        return None

    # OpenAI tool-call shape
    fn = data.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return coerce_call({"name": fn["name"], "arguments": fn.get("arguments")}, known_params)

    for nk in _NAME_KEYS:
        name = data.get(nk)
        if isinstance(name, str) and name in known_params:
            params = None
            for pk in _PARAM_KEYS:
                if pk in data:
                    params = _as_params(data[pk], name, known_params)
                    break
            else:
                params = {k: v for k, v in data.items() if k not in _NAME_KEYS}
            if params is None:
                return None
            return name, params
        if isinstance(name, str) and _PSEUDO_CALL_RE.match(name):
            parsed = _parse_pseudo_call(name, known_params)
            if parsed:
                return parsed

    if len(data) == 1:
        (key, value), = data.items()
        if key in known_params:
            params = _as_params(value, key, known_params)
            return (key, params) if params is not None else None

    return None
