"""
LLM backends for Autobot's decision-making ("the manager").

Autobot must work without a paid API key. The backends, in the order the
auto-detected chain tries them:

  openai_compat  any OpenAI-compatible endpoint you point it at
                 (AUTOBOT_LLM_BASE_URL / AUTOBOT_LLM_API_KEY / AUTOBOT_LLM_MODEL)
  groq           free tier, GROQ_API_KEY
  cerebras       free tier, CEREBRAS_API_KEY
  openrouter     OPENROUTER_API_KEY (":free" models work without credits)
  ollama         local models, nothing leaves the machine
  claude_cli     your Claude Pro subscription via `claude -p` (no API key)
  agy_cli        your Antigravity subscription via `agy -p` (no API key)
  gemini         Google AI Studio free tier (GEMINI_API_KEY) — last, because
                 outside the EU/UK the free tier may use prompts for training

Every backend offers complete_json(system, user, schema) and
complete_text(system, user). Where the backend supports it, the schema is
enforced by the provider (Claude/agy --json-schema, OpenAI-style
response_format); everywhere else the reply goes through the tolerant
parser in autobot/util/jsonx.py. Errors are raised as LLMError with an
error_class so the caller can fall through to the next backend or back off.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from autobot.util.jsonx import extract_json

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    def __init__(self, message: str, error_class: str = "other") -> None:
        super().__init__(message)
        self.error_class = error_class   # rate_limited | auth | unavailable | bad_output | other


class LLMBackend:
    name = "base"
    trains_on_data = False   # True only where the provider documents it (Gemini free tier)

    def available(self) -> bool:
        return True

    def complete_text(self, system: str, user: str, timeout: float = 180.0) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: dict, timeout: float = 180.0) -> dict:
        text = self.complete_text(
            system + "\n\nReply with ONE JSON object matching this JSON Schema, and nothing else:\n"
            + json.dumps(schema),
            user, timeout,
        )
        data = extract_json(text)
        if not isinstance(data, dict):
            raise LLMError(f"{self.name}: reply was not a JSON object: {text[:200]!r}", "bad_output")
        return data

    def describe(self) -> str:
        return self.name


# ── OpenAI-compatible HTTP endpoints ─────────────────────────────────────────

class OpenAICompatBackend(LLMBackend):
    def __init__(self, name: str, base_url: str | None, api_key: str | None, model: str,
                 trains_on_data: bool = False) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key = api_key or "not-needed"
        self.model = model
        self.trains_on_data = trains_on_data
        self._client = None
        self._json_mode = "json_schema"   # downgraded on first rejection: json_schema -> json_object -> none

    def describe(self) -> str:
        return f"{self.name} ({self.model})"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise LLMError("openai package not installed (pip install openai)", "unavailable") from e
            kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _call(self, messages: list[dict], timeout: float, response_format: dict | None) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.1,
                                  "timeout": timeout}
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise _classify_http_error(self.name, e) from e
        content = resp.choices[0].message.content if resp.choices else None
        return (content or "").strip()

    def complete_text(self, system: str, user: str, timeout: float = 180.0) -> str:
        return self._call([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          timeout, None)

    def complete_json(self, system: str, user: str, schema: dict, timeout: float = 180.0) -> dict:
        sys_prompt = (system + "\n\nReply with ONE JSON object matching this JSON Schema, and nothing else:\n"
                      + json.dumps(schema))
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        while True:
            if self._json_mode == "json_schema":
                fmt = {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema, "strict": False}}
            elif self._json_mode == "json_object":
                fmt = {"type": "json_object"}
            else:
                fmt = None
            try:
                text = self._call(messages, timeout, fmt)
                break
            except LLMError as e:
                if e.error_class == "bad_request" and self._json_mode != "none":
                    self._json_mode = "json_object" if self._json_mode == "json_schema" else "none"
                    continue
                raise
        data = extract_json(text)
        if not isinstance(data, dict):
            raise LLMError(f"{self.name}: reply was not a JSON object: {text[:200]!r}", "bad_output")
        return data


def _classify_http_error(name: str, e: Exception) -> LLMError:
    status = getattr(e, "status_code", None)
    text = f"{type(e).__name__}: {e}"
    low = text.lower()
    if status == 429 or "rate limit" in low or "quota" in low or "resource_exhausted" in low:
        return LLMError(f"{name}: {text}", "rate_limited")
    if status in (401, 403) or "api key" in low or "authentication" in low:
        return LLMError(f"{name}: {text}", "auth")
    if status == 402 or "insufficient credits" in low:
        return LLMError(f"{name}: {text}", "auth")
    if status == 400 or "response_format" in low or "json_schema" in low:
        return LLMError(f"{name}: {text}", "bad_request")
    if "connect" in low or "timed out" in low or "timeout" in low:
        return LLMError(f"{name}: {text}", "unavailable")
    return LLMError(f"{name}: {text}", "other")


# ── Subscription CLIs (no API key) ───────────────────────────────────────────

def _workspace() -> str:
    from autobot.paths import state_dir
    return str(state_dir("manager_workspace"))


class ClaudeCLIBackend(LLMBackend):
    """Claude Code in print mode with ALL tools disabled — a pure model call
    on the user's Claude subscription, schema-enforced via --json-schema."""
    name = "claude_cli"

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def describe(self) -> str:
        return f"claude_cli ({self.model or 'account default model'})"

    def available(self) -> bool:
        from autobot.integrations import claude_code_bridge
        return claude_code_bridge.is_available()

    def _run(self, prompt: str, timeout: float, schema: dict | None) -> dict:
        from autobot.integrations import claude_code_bridge
        r = claude_code_bridge.run_headless(
            prompt, cwd=_workspace(), permission_mode="plan", tools=[], json_schema=schema,
            model=self.model, no_session_persistence=True, timeout=timeout,
        )
        if not r["ok"]:
            cls = r.get("error_class") or "other"
            raise LLMError(f"claude_cli: {r['error']}", "unavailable" if cls in ("not_installed", "timeout") else cls)
        return r["data"] if isinstance(r["data"], dict) else {"result": str(r["data"])}

    def complete_text(self, system: str, user: str, timeout: float = 240.0) -> str:
        data = self._run(f"{system}\n\n---\n\n{user}", timeout, None)
        return str(data.get("result", "")).strip()

    def complete_json(self, system: str, user: str, schema: dict, timeout: float = 240.0) -> dict:
        data = self._run(f"{system}\n\n---\n\n{user}", timeout, schema)
        so = data.get("structured_output")
        if isinstance(so, dict):
            return so
        parsed = extract_json(str(data.get("result", "")))
        if isinstance(parsed, dict):
            return parsed
        raise LLMError(f"claude_cli: no structured output in reply: {str(data)[:200]}", "bad_output")


class AgyCLIBackend(LLMBackend):
    """Antigravity in print mode — the user's Antigravity subscription."""
    name = "agy_cli"

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def available(self) -> bool:
        from autobot.integrations import antigravity_bridge
        return antigravity_bridge.is_available()

    def _run(self, prompt: str, timeout: float, schema: dict | None) -> dict:
        from autobot.integrations import antigravity_bridge
        r = antigravity_bridge.run_headless(prompt, cwd=_workspace(), model=self.model,
                                            json_schema=schema, timeout=timeout)
        if not r["ok"]:
            cls = r.get("error_class") or "other"
            raise LLMError(f"agy_cli: {r['error']}", "unavailable" if cls in ("not_installed", "timeout") else cls)
        return r["data"] if isinstance(r["data"], dict) else {"response": str(r["data"])}

    def complete_text(self, system: str, user: str, timeout: float = 240.0) -> str:
        data = self._run(f"{system}\n\n---\n\n{user}", timeout, None)
        return str(data.get("response") or data.get("result") or "").strip()

    def complete_json(self, system: str, user: str, schema: dict, timeout: float = 240.0) -> dict:
        data = self._run(f"{system}\n\n---\n\n{user}", timeout, schema)
        so = data.get("structured_output")
        if isinstance(so, dict):
            return so
        parsed = extract_json(str(data.get("response") or data.get("result") or ""))
        if isinstance(parsed, dict):
            return parsed
        raise LLMError(f"agy_cli: no structured output in reply: {str(data)[:200]}", "bad_output")


# ── Fallback chain ───────────────────────────────────────────────────────────

class ChainBackend(LLMBackend):
    """Try each backend in order. A backend that is rate-limited is skipped
    for a cooldown period; auth/unavailable failures skip to the next one."""
    name = "chain"

    def __init__(self, backends: list[LLMBackend], cooldown_s: float = 900.0) -> None:
        self.backends = backends
        self.cooldown_s = cooldown_s
        self._cooldown_until: dict[str, float] = {}
        self.last_used: str | None = None

    def describe(self) -> str:
        return " -> ".join(b.describe() for b in self.backends) or "(no backends)"

    @property
    def trains_on_data(self) -> bool:  # type: ignore[override]
        return any(b.trains_on_data for b in self.backends)

    def without_training_backends(self) -> "ChainBackend":
        return ChainBackend([b for b in self.backends if not b.trains_on_data], self.cooldown_s)

    def available(self) -> bool:
        return bool(self.backends)

    def _attempt(self, fn_name: str, *args, **kwargs):
        errors: list[str] = []
        now = time.time()
        for b in self.backends:
            if self._cooldown_until.get(b.name, 0) > now:
                errors.append(f"{b.name}: cooling down after a rate limit")
                continue
            try:
                result = getattr(b, fn_name)(*args, **kwargs)
                self.last_used = b.name
                return result
            except LLMError as e:
                errors.append(str(e)[:300])
                if e.error_class == "rate_limited":
                    self._cooldown_until[b.name] = time.time() + self.cooldown_s
                logger.warning(f"LLM backend {b.name} failed ({e.error_class}); trying next. {e}")
        classes = "rate_limited" if errors and all("rate" in x or "cooling" in x for x in errors) else "unavailable"
        raise LLMError("All LLM backends failed:\n  " + "\n  ".join(errors or ["no backends configured"]), classes)

    def complete_text(self, system: str, user: str, timeout: float = 240.0) -> str:
        return self._attempt("complete_text", system, user, timeout)

    def complete_json(self, system: str, user: str, schema: dict, timeout: float = 240.0) -> dict:
        return self._attempt("complete_json", system, user, schema, timeout)


class ChatCompletionsShim:
    """Makes any LLMBackend look like `client.chat.completions.create(...)`,
    so CoreLoop (which speaks the OpenAI client shape) can run on
    claude_cli / agy_cli / a chain without changes."""

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend
        outer = self

        class _Completions:
            def create(self_inner, model: str = "", messages: list | None = None, **_: Any):
                return outer._create(messages or [])

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()

    def _create(self, messages: list[dict]):
        system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
        convo = "\n\n".join(
            f"[{m.get('role')}]\n{_text(m.get('content'))}" for m in messages if m.get("role") != "system"
        )
        text = self._backend.complete_text(system, convo)
        from types import SimpleNamespace
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")
