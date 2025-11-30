"""Mock LLM provider for integration testing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from jpscripts.providers import (
    CompletionOptions,
    CompletionResponse,
    Message,
    ProviderType,
    StreamChunk,
    TokenUsage,
)


class MockProvider:
    """Deterministic mock LLM provider for testing.

    Implements the LLMProvider protocol with pattern-based responses.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        """Initialize mock provider.

        Args:
            responses: Mapping of prompt patterns to JSON responses.
                       If prompt contains the key, return the value.
        """
        self._responses = responses or {}
        self._call_log: list[Message | None] = []
        self._default_response = (
            '{"thought_process":"done","criticism":null,'
            '"tool_call":null,"file_patch":null,"final_message":"No action needed"}'
        )

    @property
    def provider_type(self) -> ProviderType:
        """Return mock provider type."""
        return ProviderType.OPENAI

    @property
    def default_model(self) -> str:
        """Return default mock model."""
        return "mock-model"

    @property
    def available_models(self) -> tuple[str, ...]:
        """Return available mock models."""
        return ("mock-model",)

    @property
    def call_log(self) -> list[Message | None]:
        """Return log of all messages received."""
        return self._call_log.copy()

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Return deterministic response based on prompt patterns.

        Args:
            messages: Conversation history
            model: Model ID (ignored, uses mock)
            options: Completion options (ignored)

        Returns:
            CompletionResponse with matching pattern response or default
        """
        last_message = messages[-1] if messages else None
        self._call_log.append(last_message)

        prompt = last_message.content if last_message else ""

        # Find matching response by pattern
        for pattern, response in self._responses.items():
            if pattern in prompt:
                return CompletionResponse(
                    content=response,
                    model=model or self.default_model,
                    finish_reason="stop",
                    usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
                )

        # Default response
        return CompletionResponse(
            content=self._default_response,
            model=model or self.default_model,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Mock streaming - not supported.

        Raises:
            NotImplementedError: Always, as mock doesn't support streaming.
        """
        raise NotImplementedError("MockProvider does not support streaming")
        yield StreamChunk(content="")  # pragma: no cover - unreachable, needed for async generator type

    def supports_streaming(self) -> bool:
        """Return False - mock doesn't support streaming."""
        return False

    def supports_tools(self) -> bool:
        """Return True - mock supports tools."""
        return True

    def supports_json_mode(self) -> bool:
        """Return True - mock supports JSON mode."""
        return True

    def get_context_limit(self, model: str | None = None) -> int:
        """Return mock context limit."""
        return 128_000


class MockProviderWithCounter(MockProvider):
    """Mock provider that tracks call count and can change behavior."""

    def __init__(
        self,
        responses_by_call: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with per-call response list.

        Args:
            responses_by_call: List of responses to return in order.
                               After exhausted, returns last response.
            **kwargs: Passed to MockProvider
        """
        super().__init__(**kwargs)
        self._responses_by_call = responses_by_call or []
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Return number of complete() calls made."""
        return self._call_count

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Return response based on call index."""
        self._call_count += 1

        if self._responses_by_call:
            idx = min(self._call_count - 1, len(self._responses_by_call) - 1)
            response = self._responses_by_call[idx]
            self._call_log.append(messages[-1] if messages else None)
            return CompletionResponse(
                content=response,
                model=model or self.default_model,
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            )

        return await super().complete(messages, model, options)
