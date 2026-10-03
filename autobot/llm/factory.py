"""
Build the manager LLM from configuration.

    AUTOBOT_MANAGER_LLM   comma-separated backend names, tried in order,
                          e.g. "groq,claude_cli". "none" = deterministic
                          playbooks only (the butler still runs). Unset =
                          auto-detect everything available (see ORDER).

Per-backend settings (all optional unless noted):
    AUTOBOT_LLM_BASE_URL, AUTOBOT_LLM_API_KEY, AUTOBOT_LLM_MODEL   openai_compat (base URL + model required)
    GROQ_API_KEY,       AUTOBOT_GROQ_MODEL        (default llama-3.3-70b-versatile)
    CEREBRAS_API_KEY,   AUTOBOT_CEREBRAS_MODEL    (required with Cerebras)
    OPENROUTER_API_KEY, AUTOBOT_OPENROUTER_MODEL  (default: AUTOBOT_LLM_MODEL)
    AUTOBOT_OLLAMA_URL (default http://localhost:11434/v1), AUTOBOT_OLLAMA_MODEL (required)
    AUTOBOT_CLAUDE_MANAGER_MODEL  e.g. "haiku" to spare your Claude plan's limits
    AUTOBOT_AGY_MANAGER_MODEL
    GEMINI_API_KEY / GOOGLE_API_KEY, AUTOBOT_GEMINI_MODEL (default gemini-2.5-flash)
"""
from __future__ import annotations

import os
import urllib.request

from autobot.llm.backends import (
    AgyCLIBackend,
    ChainBackend,
    ClaudeCLIBackend,
    LLMBackend,
    OpenAICompatBackend,
)

ORDER = ["openai_compat", "groq", "cerebras", "openrouter", "ollama", "claude_cli", "agy_cli", "gemini"]


def _env(name: str) -> str | None:
    v = os.getenv(name, "").strip()
    return v if v and v.lower() not in ("none", "null", "undefined") else None


def _ollama_reachable(url: str) -> bool:
    root = url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/api/tags", timeout=0.8) as r:
            return r.status == 200
    except Exception:
        return False


def build_backend(name: str, explicit: bool = False) -> LLMBackend | None:
    """One backend by name, or None if it isn't configured/available.

    explicit=True (the user listed it by name) skips the cheap reachability
    probes, so a misconfiguration surfaces as a real error at call time
    instead of silently vanishing from the chain.
    """
    if name == "openai_compat":
        base, model = _env("AUTOBOT_LLM_BASE_URL"), _env("AUTOBOT_LLM_MODEL")
        return OpenAICompatBackend("openai_compat", base, _env("AUTOBOT_LLM_API_KEY"), model) if base and model else None
    if name == "groq":
        key = _env("GROQ_API_KEY")
        return OpenAICompatBackend("groq", "https://api.groq.com/openai/v1", key,
                                   _env("AUTOBOT_GROQ_MODEL") or "llama-3.3-70b-versatile") if key else None
    if name == "cerebras":
        key, model = _env("CEREBRAS_API_KEY"), _env("AUTOBOT_CEREBRAS_MODEL")
        return OpenAICompatBackend("cerebras", "https://api.cerebras.ai/v1", key, model) if key and model else None
    if name == "openrouter":
        key = _env("OPENROUTER_API_KEY")
        model = _env("AUTOBOT_OPENROUTER_MODEL") or _env("AUTOBOT_LLM_MODEL")
        return OpenAICompatBackend("openrouter", "https://openrouter.ai/api/v1", key, model) if key and model else None
    if name == "ollama":
        url = _env("AUTOBOT_OLLAMA_URL") or "http://localhost:11434/v1"
        model = _env("AUTOBOT_OLLAMA_MODEL")
        if not model or (not explicit and not _ollama_reachable(url)):
            return None
        return OpenAICompatBackend("ollama", url, "ollama", model)
    if name == "claude_cli":
        b = ClaudeCLIBackend(model=_env("AUTOBOT_CLAUDE_MANAGER_MODEL"))
        return b if (explicit or b.available()) else None
    if name == "agy_cli":
        b = AgyCLIBackend(model=_env("AUTOBOT_AGY_MANAGER_MODEL"))
        return b if (explicit or b.available()) else None
    if name == "gemini":
        key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        return OpenAICompatBackend(
            "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", key,
            _env("AUTOBOT_GEMINI_MODEL") or "gemini-2.5-flash", trains_on_data=True,
        ) if key else None
    return None


def get_manager_llm() -> ChainBackend | None:
    """The configured chain, or None when the manager should run LLM-free."""
    spec = os.getenv("AUTOBOT_MANAGER_LLM", "").strip().lower()   # raw: "none" must mean none
    if spec in ("none", "off", "false", "0"):
        return None
    names = [n.strip() for n in spec.split(",") if n.strip()] if spec else ORDER
    explicit = bool(spec)
    backends = [b for b in (build_backend(n, explicit=explicit) for n in names) if b is not None]
    return ChainBackend(backends) if backends else None
