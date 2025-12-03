"""Mock LLM provider for integration and contract testing.

Provides deterministic LLM provider implementations that satisfy the
LLMProvider protocol for testing without making real API calls.
"""

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
    Supports error simulation for testing error handling paths.

    Usage:
        # Basic usage
        provider = MockProvider(responses={"hello": "Hi there!"})

        # With error simulation
        provider = MockProvider()
        provider.simulate_error(RateLimitError("Rate limit exceeded", retry_after=60))
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        streaming_enabled: bool = False,
    ) -> None:
        """Initialize mock provider.

        Args:
            responses: Mapping of prompt patterns to JSON responses.
                       If prompt contains the key, return the value.
            streaming_enabled: If True, stream() returns chunks instead of raising.
        """
        self._responses = responses or {}
        self._call_log: list[Message | None] = []
        self._default_response = (
            '{"thought_process":"done","criticism":null,'
            '"tool_call":null,"file_patch":null,"final_message":"No action needed"}'
        )
        self._streaming_enabled = streaming_enabled
        self._simulated_error: Exception | None = None
        self._error_on_call: int | None = None  # Trigger error on specific call number
        self._call_count = 0

    def simulate_error(self, error: Exception, *, on_call: int | None = None) -> MockProvider:
        """Configure the provider to raise an error.

        Args:
            error: The exception to raise on next call
            on_call: If set, only raise on this specific call number (1-indexed)

        Returns:
            Self for chaining
        """
        self._simulated_error = error
        self._error_on_call = on_call
        return self

    def clear_error(self) -> MockProvider:
        """Clear any simulated error.

        Returns:
            Self for chaining
        """
        self._simulated_error = None
        self._error_on_call = None
        return self

    def _check_and_raise_error(self) -> None:
        """Check if an error should be raised and raise it."""
        if self._simulated_error is None:
            return

        # Check if error should trigger on specific call
        if self._error_on_call is not None and self._call_count != self._error_on_call:
            return

        error = self._simulated_error
        # Clear one-shot errors
        if self._error_on_call is not None:
            self._simulated_error = None
            self._error_on_call = None
        raise error

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

        Raises:
            ProviderError: If error simulation is configured
        """
        self._call_count += 1
        self._check_and_raise_error()

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
        """Stream a completion response chunk by chunk.

        If streaming is disabled (default), raises NotImplementedError.
        If enabled, yields chunks from the response content.

        Raises:
            NotImplementedError: If streaming_enabled is False
            ProviderError: If error simulation is configured
        """
        self._call_count += 1
        self._check_and_raise_error()

        if not self._streaming_enabled:
            raise NotImplementedError("MockProvider streaming not enabled")

        # Get the response content
        last_message = messages[-1] if messages else None
        self._call_log.append(last_message)
        prompt = last_message.content if last_message else ""

        content = self._default_response
        for pattern, response in self._responses.items():
            if pattern in prompt:
                content = response
                break

        # Yield content in chunks
        chunk_size = 10
        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            is_last = i + chunk_size >= len(content)
            yield StreamChunk(
                content=chunk,
                finish_reason="stop" if is_last else None,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50) if is_last else None,
            )

    def supports_streaming(self) -> bool:
        """Return whether streaming is enabled."""
        return self._streaming_enabled

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
    """Mock provider that returns different responses based on call order.

    Extends MockProvider to return responses from a list, useful for testing
    multi-turn conversations where different responses are needed.
    """

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
        """Return response based on call index.

        Args:
            messages: Conversation history
            model: Model ID (ignored)
            options: Completion options (ignored)

        Returns:
            Response from responses_by_call list or falls back to parent
        """
        self._call_count += 1
        self._check_and_raise_error()

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

        # Reset call count since parent will increment again
        self._call_count -= 1
        return await super().complete(messages, model, options)
