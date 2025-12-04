"""
OpenAI provider implementation.

This module provides direct integration with the OpenAI API
for GPT-4 and o1/o3 series models.

Usage:
    from jpscripts.providers.openai import OpenAIProvider

    provider = OpenAIProvider(config)
    response = await provider.complete(
        messages=[Message(role="user", content="Hello")],
        model="gpt-4o",
    )
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, cast

from jpscripts.providers import (
    AuthenticationError,
    BaseLLMProvider,
    CompletionOptions,
    CompletionResponse,
    ContentFilterError,
    ContextLengthError,
    Message,
    ModelNotFoundError,
    ProviderError,
    ProviderType,
    RateLimitError,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    register_provider,
)

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig

# Pattern to match potential API keys in error messages
_API_KEY_PATTERN = re.compile(
    r"""
    # OpenAI key pattern: sk-[base64 chars]
    sk-[A-Za-z0-9]{20,}|
    # Generic API key patterns that might appear in error messages
    (?:api[_-]?key|secret|token|password|credential)
    \s*[=:]\s*
    ['"]?[A-Za-z0-9_\-]{16,}['"]?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _redact_api_key(message: str) -> str:
    """Remove potential API keys from error messages to prevent leaking secrets."""
    # Also check the environment variable value directly
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and api_key in message:
        message = message.replace(api_key, "[REDACTED]")

    # Apply pattern-based redaction for other potential secrets
    message = _API_KEY_PATTERN.sub("[REDACTED]", message)

    return message


class _ToolFunction(Protocol):
    name: str
    arguments: str


class _ChatToolCall(Protocol):
    id: str
    function: _ToolFunction


class _CompletionMessage(Protocol):
    content: str | None
    tool_calls: list[_ChatToolCall] | None


class _CompletionChoice(Protocol):
    message: _CompletionMessage
    finish_reason: str | None


class _Usage(Protocol):
    prompt_tokens: int
    completion_tokens: int


class _CompletionResponse(Protocol):
    choices: list[_CompletionChoice]
    usage: _Usage | None
    model: str


class _Delta(Protocol):
    content: str | None


class _StreamChoice(Protocol):
    delta: _Delta | None
    finish_reason: str | None


class _StreamUsage(Protocol):
    prompt_tokens: int
    completion_tokens: int


class _StreamChunk(Protocol):
    choices: list[_StreamChoice]
    usage: _StreamUsage | None


class _CompletionsAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _ChatAPI(Protocol):
    completions: _CompletionsAPI


class OpenAIClientProtocol(Protocol):
    chat: _ChatAPI


# Model context limits (tokens)
OPENAI_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-2024-11-20": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "o1": 200_000,
    "o1-2024-12-17": 200_000,
    "o1-mini": 128_000,
    "o1-mini-2024-09-12": 128_000,
    "o3-mini": 200_000,
}

OPENAI_AVAILABLE_MODELS: tuple[str, ...] = (
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "o1-2024-12-17",
    "o1-mini-2024-09-12",
)

# Aliases to canonical model IDs
MODEL_ALIASES: dict[str, str] = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "o1": "o1-2024-12-17",
    "o1-mini": "o1-mini-2024-09-12",
}

# Models that support different features
MODELS_WITHOUT_SYSTEM_PROMPT: frozenset[str] = frozenset(
    {
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o3-mini",
    }
)

MODELS_WITHOUT_TEMPERATURE: frozenset[str] = frozenset(
    {
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o3-mini",
    }
)


def _resolve_model_id(model: str) -> str:
    """Resolve model alias to canonical ID."""
    return MODEL_ALIASES.get(model, model)


def _convert_messages_to_openai(
    messages: list[Message],
    system_prompt: str | None = None,
    model: str = "",
) -> list[dict[str, object]]:
    """Convert our Message format to OpenAI's format.

    Args:
        messages: List of messages to convert
        system_prompt: Optional system prompt to prepend
        model: Model ID (some models don't support system role)

    Returns:
        List of OpenAI-format messages
    """
    converted: list[dict[str, object]] = []
    supports_system = model not in MODELS_WITHOUT_SYSTEM_PROMPT

    # Add system prompt if provided and supported
    if system_prompt and supports_system:
        converted.append({"role": "system", "content": system_prompt})
    elif system_prompt and not supports_system:
        # For o1/o3 models, prepend system content to first user message
        pass  # Will be handled below

    system_prepend = system_prompt if (system_prompt and not supports_system) else None

    for _i, msg in enumerate(messages):
        if msg.role == "system":
            if supports_system:
                converted.append({"role": "system", "content": msg.content})
            else:
                # Merge system content into user message for o1/o3
                if system_prepend:
                    system_prepend = f"{system_prepend}\n\n{msg.content}"
                else:
                    system_prepend = msg.content
        else:
            content = msg.content
            # Prepend accumulated system content to first user message
            if system_prepend and msg.role == "user":
                content = f"[System Context]\n{system_prepend}\n\n[User Message]\n{content}"
                system_prepend = None

            message_dict: dict[str, object] = {
                "role": msg.role,
                "content": content,
            }
            if msg.name:
                message_dict["name"] = msg.name
            converted.append(message_dict)

    return converted


def _convert_tools_to_openai(
    tools: tuple[ToolDefinition, ...] | None,
) -> list[dict[str, object]] | None:
    """Convert our ToolDefinition format to OpenAI's format."""
    if not tools:
        return None

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _parse_tool_calls(
    tool_calls: list[_ChatToolCall] | None,
) -> list[ToolCall]:
    """Parse tool calls from OpenAI response."""
    if not tool_calls:
        return []

    import json

    result: list[ToolCall] = []
    for tc in tool_calls:
        try:
            arguments = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            arguments = {}

        result.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=arguments,
            )
        )
    return result


def _build_completion_params(
    model_id: str,
    messages: list[dict[str, object]],
    opts: CompletionOptions,
    *,
    stream: bool = False,
) -> dict[str, object]:
    """Build request parameters for OpenAI completion API.

    Consolidates parameter building logic shared between complete() and stream().

    Args:
        model_id: Resolved model ID (canonical, not alias)
        messages: Already-converted OpenAI format messages
        opts: Completion options
        stream: Whether this is a streaming request

    Returns:
        Dictionary of parameters ready for API call
    """
    params: dict[str, object] = {
        "model": model_id,
        "messages": messages,
    }

    if stream:
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

    # Max tokens parameter differs by model
    if opts.max_tokens:
        if model_id.startswith(("o1", "o3")):
            params["max_completion_tokens"] = opts.max_tokens
        else:
            params["max_tokens"] = opts.max_tokens

    # Temperature not supported for o1/o3 models
    if opts.temperature is not None and model_id not in MODELS_WITHOUT_TEMPERATURE:
        params["temperature"] = opts.temperature

    if opts.top_p is not None and model_id not in MODELS_WITHOUT_TEMPERATURE:
        params["top_p"] = opts.top_p

    if opts.stop_sequences:
        params["stop"] = list(opts.stop_sequences)

    # JSON mode
    if opts.json_mode:
        params["response_format"] = {"type": "json_object"}

    # Tools (not supported for o1 series)
    tools = _convert_tools_to_openai(opts.tools)
    if tools and not model_id.startswith(("o1", "o3")):
        params["tools"] = tools
        # tool_choice only for non-streaming requests
        if not stream and opts.tool_choice:
            if opts.tool_choice in ("auto", "none"):
                params["tool_choice"] = opts.tool_choice
            else:
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": opts.tool_choice},
                }

    # Reasoning effort for o1/o3 models
    if opts.reasoning_effort and model_id.startswith(("o1", "o3")):
        params["reasoning_effort"] = opts.reasoning_effort

    return params


def _extract_completion_choice(
    response: _CompletionResponse,
) -> tuple[str, list[ToolCall], str | None]:
    """Extract content, tool calls, and finish reason from first choice.

    Args:
        response: The completion response object

    Returns:
        Tuple of (content, tool_calls, finish_reason)
    """
    choice = response.choices[0] if response.choices else None
    if not choice:
        return "", [], None

    content = choice.message.content or ""
    tool_calls = _parse_tool_calls(choice.message.tool_calls)
    finish_reason = choice.finish_reason
    return content, tool_calls, finish_reason


def _parse_stream_chunk(chunk: _StreamChunk) -> tuple[str, str | None]:
    """Parse content and finish reason from a stream chunk.

    Args:
        chunk: A single stream chunk with choices

    Returns:
        Tuple of (content_delta, finish_reason)
    """
    choice = chunk.choices[0]
    delta = choice.delta

    content = ""
    if delta and delta.content:
        content = delta.content

    return content, choice.finish_reason


@register_provider(ProviderType.OPENAI)
class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation.

    Requires the `openai` package to be installed.
    API key is read from OPENAI_API_KEY environment variable.
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._client: OpenAIClientProtocol | None = None

    def _get_client(self) -> OpenAIClientProtocol:
        """Lazy-initialize the OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            import openai  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ProviderError(
                "openai package not installed. Install with: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AuthenticationError("OPENAI_API_KEY environment variable not set")

        client = openai.AsyncOpenAI(api_key=api_key)
        self._client = cast(OpenAIClientProtocol, client)
        return self._client

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    @property
    def default_model(self) -> str:
        return "gpt-4o-2024-11-20"

    @property
    def available_models(self) -> tuple[str, ...]:
        return OPENAI_AVAILABLE_MODELS

    def get_context_limit(self, model: str | None = None) -> int:
        model_id = model or self.default_model
        resolved = _resolve_model_id(model_id)
        return OPENAI_CONTEXT_LIMITS.get(resolved, 128_000)

    def supports_tools(self) -> bool:
        """Most OpenAI models support tools, but not o1 series (yet)."""
        return True

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Send a completion request to OpenAI."""
        client = self._get_client()
        opts = options or CompletionOptions()

        model_id = _resolve_model_id(model or self.default_model)
        converted_messages = _convert_messages_to_openai(messages, opts.system_prompt, model_id)
        params = _build_completion_params(model_id, converted_messages, opts)

        try:
            response_obj = await client.chat.completions.create(**params)
        except Exception as exc:
            self._handle_api_error(exc)
            raise AssertionError("unreachable")

        response = cast(_CompletionResponse, response_obj)
        content, tool_calls, finish_reason = _extract_completion_choice(response)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return CompletionResponse(
            content=content,
            model=response.model,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=response,
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion response from OpenAI."""
        client = self._get_client()
        opts = options or CompletionOptions()

        model_id = _resolve_model_id(model or self.default_model)
        converted_messages = _convert_messages_to_openai(messages, opts.system_prompt, model_id)
        params = _build_completion_params(model_id, converted_messages, opts, stream=True)

        try:
            stream_obj = await client.chat.completions.create(**params)
            stream = cast(AsyncIterator[_StreamChunk], stream_obj)
            async for chunk in stream:
                if not chunk.choices:
                    # Final chunk with usage only
                    if chunk.usage:
                        yield StreamChunk(
                            content="",
                            usage=TokenUsage(
                                prompt_tokens=chunk.usage.prompt_tokens,
                                completion_tokens=chunk.usage.completion_tokens,
                            ),
                        )
                    continue

                content, finish_reason = _parse_stream_chunk(chunk)
                yield StreamChunk(content=content, finish_reason=finish_reason)
        except Exception as exc:
            self._handle_api_error(exc)

    def _handle_api_error(self, exc: Exception) -> None:
        """Convert OpenAI exceptions to our error types.

        All error messages are redacted to prevent API key leakage.
        """
        try:
            import openai  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ProviderError(_redact_api_key(str(exc))) from exc

        # Redact the error message to prevent API key leakage
        safe_msg = _redact_api_key(str(exc))

        if isinstance(exc, openai.AuthenticationError):
            raise AuthenticationError(safe_msg) from exc
        if isinstance(exc, openai.RateLimitError):
            raise RateLimitError(safe_msg) from exc
        if isinstance(exc, openai.NotFoundError):
            raise ModelNotFoundError(safe_msg) from exc
        if isinstance(exc, openai.BadRequestError):
            msg_lower = safe_msg.lower()
            if "context" in msg_lower or "token" in msg_lower or "length" in msg_lower:
                raise ContextLengthError(safe_msg) from exc
            if "content" in msg_lower or "filter" in msg_lower or "policy" in msg_lower:
                raise ContentFilterError(safe_msg) from exc
            raise ProviderError(safe_msg) from exc
        if isinstance(exc, openai.APIError):
            raise ProviderError(safe_msg) from exc

        raise ProviderError(safe_msg) from exc


__all__ = [
    "OPENAI_AVAILABLE_MODELS",
    "OPENAI_CONTEXT_LIMITS",
    "OpenAIProvider",
]
