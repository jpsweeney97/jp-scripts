"""
LLM Provider abstraction for jp-scripts.

This module provides a unified interface for interacting with different LLM
backends (Anthropic Claude, OpenAI GPT, Codex CLI, etc.) while maintaining
full feature parity across providers.

Usage:
    from jpscripts.providers import get_provider, LLMProvider

    # Get provider based on config/model
    provider = get_provider(config, model_id="claude-opus-4-5")

    # Send a completion request
    response = await provider.complete(
        messages=[Message(role="user", content="Hello")],
        temperature=0.7,
    )

    # Stream responses
    async for chunk in provider.stream(messages):
        print(chunk.content, end="")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig


class ProviderType(Enum):
    """Supported LLM provider backends."""

    ANTHROPIC = auto()
    OPENAI = auto()
    CODEX = auto()  # Codex CLI wrapper


@dataclass(frozen=True, slots=True)
class Message:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "system"
    content: str
    name: str | None = None  # Optional name for multi-agent scenarios


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class CompletionResponse:
    """Response from an LLM completion request."""

    content: str
    model: str
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    raw_response: Any = None  # Provider-specific response object


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage for a completion request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.total_tokens is None:
            object.__setattr__(
                self, "total_tokens", self.prompt_tokens + self.completion_tokens
            )


@dataclass(slots=True)
class StreamChunk:
    """A chunk of streamed response content."""

    content: str
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class CompletionOptions:
    """Options for completion requests.

    These options are normalized across providers - each provider
    implementation handles mapping to provider-specific parameters.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: tuple[str, ...] | None = None
    tools: tuple[ToolDefinition, ...] | None = None
    tool_choice: str | None = None  # "auto", "none", or specific tool name
    reasoning_effort: str | None = None  # For models that support it
    json_mode: bool = False  # Request JSON output
    system_prompt: str | None = None


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available."""

    pass


class ContentFilterError(ProviderError):
    """Raised when content is blocked by safety filters."""

    pass


class ContextLengthError(ProviderError):
    """Raised when input exceeds model's context length."""

    pass


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers.

    All provider implementations must satisfy this protocol to ensure
    consistent behavior across different backends.
    """

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...

    @property
    def default_model(self) -> str:
        """Return the default model ID for this provider."""
        ...

    @property
    def available_models(self) -> tuple[str, ...]:
        """Return tuple of available model IDs."""
        ...

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Send a completion request and return the full response.

        Args:
            messages: The conversation history
            model: Model ID (defaults to provider's default)
            options: Completion options

        Returns:
            The completion response

        Raises:
            ProviderError: On API errors
        """
        ...

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion response chunk by chunk.

        Args:
            messages: The conversation history
            model: Model ID (defaults to provider's default)
            options: Completion options

        Yields:
            Stream chunks as they arrive

        Raises:
            ProviderError: On API errors
        """
        ...

    def supports_streaming(self) -> bool:
        """Return True if this provider supports streaming."""
        ...

    def supports_tools(self) -> bool:
        """Return True if this provider supports tool/function calling."""
        ...

    def supports_json_mode(self) -> bool:
        """Return True if this provider supports native JSON mode."""
        ...

    def get_context_limit(self, model: str | None = None) -> int:
        """Return the context window size for the given model."""
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Provides common functionality and enforces the LLMProvider protocol.
    Subclasses should implement the abstract methods.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model ID."""
        ...

    @property
    @abstractmethod
    def available_models(self) -> tuple[str, ...]:
        """Return available model IDs."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Send a completion request."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion response."""
        ...

    def supports_streaming(self) -> bool:
        """Default: streaming supported."""
        return True

    def supports_tools(self) -> bool:
        """Default: tools supported."""
        return True

    def supports_json_mode(self) -> bool:
        """Default: JSON mode supported."""
        return True

    @abstractmethod
    def get_context_limit(self, model: str | None = None) -> int:
        """Return context limit for model."""
        ...


# Model ID to provider type mapping
MODEL_PROVIDER_MAP: dict[str, ProviderType] = {
    # Anthropic Claude models
    "claude-opus-4-5": ProviderType.ANTHROPIC,
    "claude-opus-4-5-20251101": ProviderType.ANTHROPIC,
    "claude-sonnet-4-5": ProviderType.ANTHROPIC,
    "claude-sonnet-4-5-20250929": ProviderType.ANTHROPIC,
    "claude-sonnet-4": ProviderType.ANTHROPIC,
    "claude-sonnet-4-20250514": ProviderType.ANTHROPIC,
    "claude-haiku-3-5": ProviderType.ANTHROPIC,
    "claude-3-5-haiku-20241022": ProviderType.ANTHROPIC,
    "claude-3-opus": ProviderType.ANTHROPIC,
    "claude-3-opus-20240229": ProviderType.ANTHROPIC,
    "claude-3-sonnet": ProviderType.ANTHROPIC,
    "claude-3-sonnet-20240229": ProviderType.ANTHROPIC,
    "claude-3-haiku": ProviderType.ANTHROPIC,
    "claude-3-haiku-20240307": ProviderType.ANTHROPIC,
    # OpenAI models
    "gpt-4-turbo": ProviderType.OPENAI,
    "gpt-4-turbo-preview": ProviderType.OPENAI,
    "gpt-4o": ProviderType.OPENAI,
    "gpt-4o-2024-11-20": ProviderType.OPENAI,
    "gpt-4o-mini": ProviderType.OPENAI,
    "gpt-4o-mini-2024-07-18": ProviderType.OPENAI,
    "o1": ProviderType.OPENAI,
    "o1-2024-12-17": ProviderType.OPENAI,
    "o1-mini": ProviderType.OPENAI,
    "o1-mini-2024-09-12": ProviderType.OPENAI,
    "o3-mini": ProviderType.OPENAI,
}


def infer_provider_type(model_id: str) -> ProviderType:
    """Infer the provider type from a model ID.

    Args:
        model_id: The model identifier

    Returns:
        The inferred provider type

    Raises:
        ModelNotFoundError: If provider cannot be inferred
    """
    # Direct lookup
    if model_id in MODEL_PROVIDER_MAP:
        return MODEL_PROVIDER_MAP[model_id]

    # Prefix-based inference
    model_lower = model_id.lower()
    if model_lower.startswith("claude"):
        return ProviderType.ANTHROPIC
    if model_lower.startswith(("gpt-", "o1", "o3")):
        return ProviderType.OPENAI

    raise ModelNotFoundError(f"Cannot infer provider for model: {model_id}")


__all__ = [
    # Types
    "ProviderType",
    "Message",
    "ToolDefinition",
    "ToolCall",
    "CompletionResponse",
    "TokenUsage",
    "StreamChunk",
    "CompletionOptions",
    # Errors
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ContentFilterError",
    "ContextLengthError",
    # Protocol and base
    "LLMProvider",
    "BaseLLMProvider",
    # Helpers
    "MODEL_PROVIDER_MAP",
    "infer_provider_type",
]
