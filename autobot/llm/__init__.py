"""LLM backends for Autobot's manager — see backends.py and factory.py."""
from autobot.llm.backends import (  # noqa: F401
    AgyCLIBackend,
    ChainBackend,
    ChatCompletionsShim,
    ClaudeCLIBackend,
    LLMBackend,
    LLMError,
    OpenAICompatBackend,
)
from autobot.llm.factory import build_backend, get_manager_llm  # noqa: F401
