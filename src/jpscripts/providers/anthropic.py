"""
Anthropic Claude provider implementation.

This module provides direct integration with the Anthropic API
for Claude models (Opus, Sonnet, Haiku).

Usage:
    from jpscripts.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(config)
    response = await provider.complete(
        messages=[Message(role="user", content="Hello")],
        model="claude-opus-4-5",
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
)

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig

# Pattern to match potential API keys in error messages
_API_KEY_PATTERN = re.compile(
    r"""
    # Anthropic key pattern: sk-ant-[base64 chars]
    sk-ant-[A-Za-z0-9_\-]{20,}|
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
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key and api_key in message:
        message = message.replace(api_key, "[REDACTED]")

    # Apply pattern-based redaction for other potential secrets
    message = _API_KEY_PATTERN.sub("[REDACTED]", message)

    return message


class _ContentBlock(Protocol):
    type: str
    id: str | None
    name: str | None
    input: object | None
    text: str | None


class _Usage(Protocol):
    input_tokens: int
    output_tokens: int


class _MessageResponse(Protocol):
    content: list[_ContentBlock]
    model: str
    stop_reason: str | None
    usage: _Usage | None


class _ContentDelta(Protocol):
    text: str | None


class _StreamEvent(Protocol):
    type: str
    delta: _ContentDelta | None


class _StreamSession(Protocol):
    """Protocol for streaming session (Python 3.11 compatible)."""

    def __aiter__(self) -> AsyncIterator[_StreamEvent]: ...
    async def __anext__(self) -> _StreamEvent: ...
    async def get_final_message(self) -> _MessageResponse: ...


class _MessagesStream(Protocol):
    """Protocol for messages stream context manager (Python 3.11 compatible)."""

    async def __aenter__(self) -> _StreamSession: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None: ...


class _MessagesAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...

    def stream(self, **kwargs: object) -> _MessagesStream: ...


class AnthropicClientProtocol(Protocol):
    messages: _MessagesAPI


# Model context limits (tokens)
ANTHROPIC_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-5": 200_000,
    "claude-opus-4-5-20251101": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-haiku-3-5": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3-haiku-20240307": 200_000,
}

ANTHROPIC_AVAILABLE_MODELS: tuple[str, ...] = (
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
)

# Aliases to canonical model IDs
MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-haiku-3-5": "claude-3-5-haiku-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


def _resolve_model_id(model: str) -> str:
    """Resolve model alias to canonical ID."""
    return MODEL_ALIASES.get(model, model)


def _convert_messages_to_anthropic(
    messages: list[Message],
    system_prompt: str | None = None,
) -> tuple[str | None, list[dict[str, object]]]:
    """Convert our Message format to Anthropic's format.

    Anthropic expects system prompt separate from messages,
    and messages must alternate between user and assistant.

    Returns:
        Tuple of (system_prompt, messages)
    """
    system = system_prompt
    converted: list[dict[str, object]] = []

    for msg in messages:
        if msg.role == "system":
            # Anthropic uses a separate system parameter
            system = f"{system}\n\n{msg.content}" if system else msg.content
        else:
            role = "user" if msg.role == "user" else "assistant"
            converted.append({"role": role, "content": msg.content})

    # Anthropic requires messages to start with user and alternate
    # If first message is assistant, prepend empty user message
    if converted and converted[0]["role"] == "assistant":
        converted.insert(0, {"role": "user", "content": "(continuing)"})

    return system, converted


def _convert_tools_to_anthropic(
    tools: tuple[ToolDefinition, ...] | None,
) -> list[dict[str, object]] | None:
    """Convert our ToolDefinition format to Anthropic's format."""
    if not tools:
        return None

    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]


def _parse_tool_calls(content_blocks: list[_ContentBlock]) -> list[ToolCall]:
    """Parse tool use blocks from Anthropic response."""
    tool_calls: list[ToolCall] = []
    for block in content_blocks:
        if block.type == "tool_use":
            call_id = block.id or ""
            call_name = block.name or ""
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=call_name,
                    arguments=dict(block.input) if isinstance(block.input, dict) else {},
                )
            )
    return tool_calls


def _extract_text_content(content_blocks: list[_ContentBlock]) -> str:
    """Extract text content from Anthropic response blocks."""
    text_parts: list[str] = []
    for block in content_blocks:
        if block.type == "text" and block.text:
            text_parts.append(block.text)
    return "".join(text_parts)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation.

    Requires the `anthropic` package to be installed.
    API key is read from ANTHROPIC_API_KEY environment variable
    or can be passed via config.
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._client: AnthropicClientProtocol | None = None

    def _get_client(self) -> AnthropicClientProtocol:
        """Lazy-initialize the Anthropic client."""
        if self._client is not None:
            return self._client

        try:
            import anthropic  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ProviderError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError("ANTHROPIC_API_KEY environment variable not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        self._client = cast(AnthropicClientProtocol, client)
        return self._client

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-5-20250929"

    @property
    def available_models(self) -> tuple[str, ...]:
        return ANTHROPIC_AVAILABLE_MODELS

    def get_context_limit(self, model: str | None = None) -> int:
        model_id = model or self.default_model
        resolved = _resolve_model_id(model_id)
        return ANTHROPIC_CONTEXT_LIMITS.get(resolved, 200_000)

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Send a completion request to Anthropic."""
        client = self._get_client()
        opts = options or CompletionOptions()

        model_id = _resolve_model_id(model or self.default_model)
        system, converted_messages = _convert_messages_to_anthropic(messages, opts.system_prompt)

        # Build request parameters
        params: dict[str, object] = {
            "model": model_id,
            "messages": converted_messages,
            "max_tokens": opts.max_tokens or 4096,
        }

        if system:
            params["system"] = system

        if opts.temperature is not None:
            params["temperature"] = opts.temperature

        if opts.top_p is not None:
            params["top_p"] = opts.top_p

        if opts.stop_sequences:
            params["stop_sequences"] = list(opts.stop_sequences)

        tools = _convert_tools_to_anthropic(opts.tools)
        if tools:
            params["tools"] = tools
            if opts.tool_choice:
                if opts.tool_choice == "auto":
                    params["tool_choice"] = {"type": "auto"}
                elif opts.tool_choice == "none":
                    params["tool_choice"] = {"type": "none"}
                else:
                    params["tool_choice"] = {"type": "tool", "name": opts.tool_choice}

        try:
            response_obj = await client.messages.create(**params)
        except Exception as exc:
            self._handle_api_error(exc)
            raise AssertionError("unreachable")

        response = cast(_MessageResponse, response_obj)

        # Parse response
        content_blocks = response.content
        text_content = _extract_text_content(content_blocks)
        tool_calls = _parse_tool_calls(content_blocks)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )

        return CompletionResponse(
            content=text_content,
            model=response.model,
            finish_reason=response.stop_reason,
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
        """Stream a completion response from Anthropic."""
        client = self._get_client()
        opts = options or CompletionOptions()

        model_id = _resolve_model_id(model or self.default_model)
        system, converted_messages = _convert_messages_to_anthropic(messages, opts.system_prompt)

        params: dict[str, object] = {
            "model": model_id,
            "messages": converted_messages,
            "max_tokens": opts.max_tokens or 4096,
        }

        if system:
            params["system"] = system

        if opts.temperature is not None:
            params["temperature"] = opts.temperature

        if opts.top_p is not None:
            params["top_p"] = opts.top_p

        if opts.stop_sequences:
            params["stop_sequences"] = list(opts.stop_sequences)

        tools = _convert_tools_to_anthropic(opts.tools)
        if tools:
            params["tools"] = tools

        try:
            stream_ctx = client.messages.stream(**params)
            async with stream_ctx as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta and delta.text:
                            yield StreamChunk(content=delta.text)
                    elif event.type == "message_stop":
                        # Final chunk with usage if available
                        final_message = await stream.get_final_message()
                        usage = None
                        if final_message.usage:
                            usage = TokenUsage(
                                prompt_tokens=final_message.usage.input_tokens,
                                completion_tokens=final_message.usage.output_tokens,
                            )
                        yield StreamChunk(
                            content="",
                            finish_reason=final_message.stop_reason,
                            usage=usage,
                        )
        except Exception as exc:
            self._handle_api_error(exc)

    def _handle_api_error(self, exc: Exception) -> None:
        """Convert Anthropic exceptions to our error types.

        All error messages are redacted to prevent API key leakage.
        """
        try:
            import anthropic  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ProviderError(_redact_api_key(str(exc))) from exc

        # Redact the error message to prevent API key leakage
        safe_msg = _redact_api_key(str(exc))

        if isinstance(exc, anthropic.AuthenticationError):
            raise AuthenticationError(safe_msg) from exc
        if isinstance(exc, anthropic.RateLimitError):
            raise RateLimitError(safe_msg) from exc
        if isinstance(exc, anthropic.NotFoundError):
            raise ModelNotFoundError(safe_msg) from exc
        if isinstance(exc, anthropic.BadRequestError):
            msg_lower = safe_msg.lower()
            if "context" in msg_lower or "token" in msg_lower:
                raise ContextLengthError(safe_msg) from exc
            if "content" in msg_lower or "filter" in msg_lower or "safety" in msg_lower:
                raise ContentFilterError(safe_msg) from exc
            raise ProviderError(safe_msg) from exc
        if isinstance(exc, anthropic.APIError):
            raise ProviderError(safe_msg) from exc

        raise ProviderError(safe_msg) from exc


__all__ = [
    "ANTHROPIC_AVAILABLE_MODELS",
    "ANTHROPIC_CONTEXT_LIMITS",
    "AnthropicProvider",
]
