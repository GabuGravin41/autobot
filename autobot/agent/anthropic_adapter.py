"""
Anthropic OpenAI Adapter for Autobot.

Allows Autobot to use a direct ANTHROPIC_API_KEY as a drop-in replacement for
OPENROUTER_API_KEY or OPENAI_API_KEY.

Translates OpenAI-style `chat.completions.create(...)` requests into
Anthropic's native `messages.create(...)` format, handling system prompts,
vision/image attachments, JSON response parsing, and max_tokens.
"""
import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional


class StructMessage:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


class StructChoice:
    def __init__(self, content: str):
        self.message = StructMessage(content)
        self.finish_reason = "stop"


class StructCompletionResponse:
    def __init__(self, content: str, model: str = "claude-sonnet-5"):
        self.choices = [StructChoice(content)]
        self.id = "msg_anthropic_adapter"
        self.model = model


class AnthropicChatCompletions:
    def __init__(self, anthropic_client: Any):
        self.client = anthropic_client

    def create(
        self,
        model: str = "claude-sonnet-5",
        messages: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> StructCompletionResponse:
        messages = messages or []

        # If model is an OpenRouter or OpenAI model name, map to default Claude model
        if not model or "claude" not in model.lower():
            model = "claude-sonnet-5"

        # Separate system messages from user/assistant conversation
        system_content = ""
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_content += ("\n" if system_content else "") + content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_content += ("\n" if system_content else "") + item.get("text", "")
                continue

            # Map OpenAI message role to Anthropic role
            anth_role = "assistant" if role in ("assistant", "model") else "user"

            # Parse content (string or list of text/image parts)
            if isinstance(content, str):
                anthropic_messages.append({"role": anth_role, "content": content})
            elif isinstance(content, list):
                anth_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    p_type = part.get("type")
                    if p_type == "text":
                        anth_parts.append({"type": "text", "text": part.get("text", "")})
                    elif p_type == "image_url":
                        url_data = part.get("image_url", {}).get("url", "")
                        # Handle data:image/png;base64,... URLs
                        if url_data.startswith("data:"):
                            match = re.match(r"data:(image/\w+);base64,(.+)", url_data)
                            if match:
                                media_type, b64_data = match.groups()
                                anth_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data
                                    }
                                })
                        elif url_data:
                            anth_parts.append({"type": "text", "text": f"[Image URL: {url_data}]"})

                if anth_parts:
                    anthropic_messages.append({"role": anth_role, "content": anth_parts})

        # Reinforce strict-JSON output every call. Previously this only fired
        # when "json" was absent from system_content — but the caller's own
        # system prompt documents the JSON schema, so it always contains the
        # word "json" and this reinforcement never actually ran. Observed
        # live failures without it: prose before the JSON block, and
        # markdown-bold section headers ("**thinking:** ...") replacing JSON
        # entirely with only a stray partial object embedded inside.
        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                system_content += (
                    "\n\nIMPORTANT: Your ENTIRE reply must be exactly one valid JSON "
                    "object — nothing before it, nothing after it. No prose lead-in, "
                    "no markdown code fences, no bold-header formatting standing in "
                    "for JSON. Every field the schema requires must be present; "
                    "omitting a required field is not acceptable."
                )

        # Call Anthropic API
        kwargs_call = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system_content:
            kwargs_call["system"] = system_content
        # Claude Sonnet 5 (and the Opus 4.6+/Fable 5 family) reject any
        # non-default temperature/top_p/top_k with a 400. This adapter has
        # no reliable way to know whether a caller-supplied `temperature`
        # differs from the model's own default, so it's never forwarded —
        # steer behavior through prompting instead.

        resp = self.client.messages.create(**kwargs_call)

        # Extract text content from Anthropic response
        out_text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                out_text += block.text

        return StructCompletionResponse(content=out_text, model=model)


class AnthropicOpenAIAdapter:
    """Wrapper that acts like an OpenAI client while directing calls to Anthropic API."""

    def __init__(self, api_key: str):
        import anthropic
        self._raw_client = anthropic.Anthropic(api_key=api_key)

        class ChatWrapper:
            def __init__(self, adapter_self):
                self.completions = AnthropicChatCompletions(adapter_self._raw_client)

        self.chat = ChatWrapper(self)


def get_anthropic_llm_client(api_key: Optional[str] = None) -> Optional[AnthropicOpenAIAdapter]:
    """Get an Anthropic LLM client if ANTHROPIC_API_KEY is available."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        return AnthropicOpenAIAdapter(api_key=key)
    except Exception as e:
        print(f"⚠️ Error creating Anthropic adapter: {e}")
        return None
