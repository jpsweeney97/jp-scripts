"""Contract tests for LLM provider implementations.

These tests verify that all providers satisfy the LLMProvider protocol
with consistent behavior. The MockProvider is used for automated testing,
while real providers can be tested manually with API keys.
"""

from __future__ import annotations

import pytest

from jpscripts.providers import (
    AuthenticationError,
    CompletionOptions,
    CompletionResponse,
    ContentFilterError,
    ContextLengthError,
    LLMProvider,
    Message,
    ProviderType,
    RateLimitError,
    StreamChunk,
    TokenUsage,
)
from tests.mocks.mock_provider import MockProvider


class TestLLMProviderProtocol:
    """Test that MockProvider satisfies the LLMProvider protocol."""

    def test_mock_provider_is_llm_provider(self) -> None:
        """Verify MockProvider satisfies the LLMProvider protocol."""
        provider = MockProvider()
        assert isinstance(provider, LLMProvider)

    def test_mock_provider_with_streaming_is_llm_provider(self) -> None:
        """Verify MockProvider with streaming enabled satisfies protocol."""
        provider = MockProvider(streaming_enabled=True)
        assert isinstance(provider, LLMProvider)


class TestProviderProperties:
    """Test required provider properties."""

    def test_provider_type(self) -> None:
        """Provider must return a valid ProviderType."""
        provider = MockProvider()
        assert isinstance(provider.provider_type, ProviderType)

    def test_default_model(self) -> None:
        """Provider must return a non-empty default model string."""
        provider = MockProvider()
        model = provider.default_model
        assert isinstance(model, str)
        assert len(model) > 0

    def test_available_models(self) -> None:
        """Provider must return a tuple of available models."""
        provider = MockProvider()
        models = provider.available_models
        assert isinstance(models, tuple)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_context_limit(self) -> None:
        """Provider must return a positive context limit."""
        provider = MockProvider()
        limit = provider.get_context_limit()
        assert isinstance(limit, int)
        assert limit > 0

    def test_context_limit_with_model(self) -> None:
        """Context limit can be queried for specific model."""
        provider = MockProvider()
        limit = provider.get_context_limit(model="mock-model")
        assert isinstance(limit, int)
        assert limit > 0


class TestBasicCompletion:
    """Test basic completion functionality."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self) -> None:
        """complete() must return a CompletionResponse."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]

        response = await provider.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_complete_with_options(self) -> None:
        """complete() must accept CompletionOptions."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]
        options = CompletionOptions(temperature=0.5, max_tokens=100)

        response = await provider.complete(messages, options=options)

        assert isinstance(response, CompletionResponse)

    @pytest.mark.asyncio
    async def test_complete_with_model(self) -> None:
        """complete() must accept explicit model parameter."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]

        response = await provider.complete(messages, model="mock-model")

        assert isinstance(response, CompletionResponse)
        assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_complete_returns_usage(self) -> None:
        """complete() should return token usage information."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]

        response = await provider.complete(messages)

        assert response.usage is not None
        assert isinstance(response.usage, TokenUsage)
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_complete_returns_finish_reason(self) -> None:
        """complete() should return a finish reason."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]

        response = await provider.complete(messages)

        assert response.finish_reason is not None
        assert isinstance(response.finish_reason, str)

    @pytest.mark.asyncio
    async def test_complete_with_pattern_matching(self) -> None:
        """MockProvider should match response patterns."""
        provider = MockProvider(responses={"test": "matched response"})
        messages = [Message(role="user", content="this is a test")]

        response = await provider.complete(messages)

        assert response.content == "matched response"


class TestStreaming:
    """Test streaming functionality."""

    def test_supports_streaming_returns_bool(self) -> None:
        """supports_streaming() must return a boolean."""
        provider = MockProvider()
        assert isinstance(provider.supports_streaming(), bool)

    def test_supports_streaming_disabled_by_default(self) -> None:
        """MockProvider has streaming disabled by default."""
        provider = MockProvider()
        assert provider.supports_streaming() is False

    def test_supports_streaming_enabled(self) -> None:
        """MockProvider can have streaming enabled."""
        provider = MockProvider(streaming_enabled=True)
        assert provider.supports_streaming() is True

    @pytest.mark.asyncio
    async def test_stream_raises_when_disabled(self) -> None:
        """stream() should raise when streaming is not supported."""
        provider = MockProvider(streaming_enabled=False)
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(NotImplementedError):
            async for _ in provider.stream(messages):
                pass

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self) -> None:
        """stream() should yield StreamChunk objects when enabled."""
        provider = MockProvider(
            streaming_enabled=True,
            responses={"hello": "Hi there!"},
        )
        messages = [Message(role="user", content="hello")]

        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(c, StreamChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_chunks_have_content(self) -> None:
        """StreamChunks should have content."""
        provider = MockProvider(
            streaming_enabled=True,
            responses={"hello": "Hi there!"},
        )
        messages = [Message(role="user", content="hello")]

        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        content = "".join(c.content for c in chunks)
        assert content == "Hi there!"

    @pytest.mark.asyncio
    async def test_stream_last_chunk_has_finish_reason(self) -> None:
        """Last StreamChunk should have finish_reason."""
        provider = MockProvider(
            streaming_enabled=True,
            responses={"hello": "Hi!"},
        )
        messages = [Message(role="user", content="hello")]

        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert chunks[-1].finish_reason is not None


class TestCapabilities:
    """Test provider capability methods."""

    def test_supports_tools(self) -> None:
        """supports_tools() must return a boolean."""
        provider = MockProvider()
        assert isinstance(provider.supports_tools(), bool)

    def test_supports_json_mode(self) -> None:
        """supports_json_mode() must return a boolean."""
        provider = MockProvider()
        assert isinstance(provider.supports_json_mode(), bool)


class TestErrorHandling:
    """Test error simulation and handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        """Provider should raise RateLimitError when simulated."""
        provider = MockProvider()
        provider.simulate_error(RateLimitError("Rate limit exceeded", retry_after=60))
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(RateLimitError) as exc_info:
            await provider.complete(messages)

        assert exc_info.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_authentication_error(self) -> None:
        """Provider should raise AuthenticationError when simulated."""
        provider = MockProvider()
        provider.simulate_error(AuthenticationError("Invalid API key"))
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(AuthenticationError):
            await provider.complete(messages)

    @pytest.mark.asyncio
    async def test_content_filter_error(self) -> None:
        """Provider should raise ContentFilterError when simulated."""
        provider = MockProvider()
        provider.simulate_error(ContentFilterError("Content blocked"))
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ContentFilterError):
            await provider.complete(messages)

    @pytest.mark.asyncio
    async def test_context_length_error(self) -> None:
        """Provider should raise ContextLengthError when simulated."""
        provider = MockProvider()
        provider.simulate_error(ContextLengthError("Input too long"))
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ContextLengthError):
            await provider.complete(messages)

    @pytest.mark.asyncio
    async def test_error_on_specific_call(self) -> None:
        """Error can be triggered on specific call number."""
        provider = MockProvider()
        provider.simulate_error(RateLimitError("Rate limit"), on_call=2)
        messages = [Message(role="user", content="Hello")]

        # First call should succeed
        response1 = await provider.complete(messages)
        assert isinstance(response1, CompletionResponse)

        # Second call should raise
        with pytest.raises(RateLimitError):
            await provider.complete(messages)

        # Third call should succeed (error was cleared)
        response3 = await provider.complete(messages)
        assert isinstance(response3, CompletionResponse)

    @pytest.mark.asyncio
    async def test_clear_error(self) -> None:
        """clear_error() should remove simulated error."""
        provider = MockProvider()
        provider.simulate_error(RateLimitError("Rate limit"))
        provider.clear_error()
        messages = [Message(role="user", content="Hello")]

        response = await provider.complete(messages)
        assert isinstance(response, CompletionResponse)

    @pytest.mark.asyncio
    async def test_stream_error_handling(self) -> None:
        """Stream should raise errors when simulated."""
        provider = MockProvider(streaming_enabled=True)
        provider.simulate_error(RateLimitError("Rate limit"))
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(RateLimitError):
            async for _ in provider.stream(messages):
                pass


class TestCallTracking:
    """Test call logging and tracking functionality."""

    @pytest.mark.asyncio
    async def test_call_log_tracks_messages(self) -> None:
        """call_log should track messages received."""
        provider = MockProvider()
        msg1 = Message(role="user", content="First")
        msg2 = Message(role="user", content="Second")

        await provider.complete([msg1])
        await provider.complete([msg2])

        log = provider.call_log
        assert len(log) == 2
        assert log[0] == msg1
        assert log[1] == msg2

    @pytest.mark.asyncio
    async def test_call_count_tracks_calls(self) -> None:
        """MockProvider should track call count."""
        provider = MockProvider()
        messages = [Message(role="user", content="Hello")]

        await provider.complete(messages)
        await provider.complete(messages)
        await provider.complete(messages)

        assert provider._call_count == 3


class TestMockProviderWithCounter:
    """Test MockProviderWithCounter functionality."""

    @pytest.mark.asyncio
    async def test_returns_responses_in_order(self) -> None:
        """Should return responses from list in order."""
        from tests.mocks.mock_provider import MockProviderWithCounter

        provider = MockProviderWithCounter(responses_by_call=["first", "second", "third"])
        messages = [Message(role="user", content="Hello")]

        r1 = await provider.complete(messages)
        r2 = await provider.complete(messages)
        r3 = await provider.complete(messages)

        assert r1.content == "first"
        assert r2.content == "second"
        assert r3.content == "third"

    @pytest.mark.asyncio
    async def test_repeats_last_response(self) -> None:
        """Should repeat last response after list exhausted."""
        from tests.mocks.mock_provider import MockProviderWithCounter

        provider = MockProviderWithCounter(responses_by_call=["first", "last"])
        messages = [Message(role="user", content="Hello")]

        await provider.complete(messages)
        await provider.complete(messages)
        r3 = await provider.complete(messages)
        r4 = await provider.complete(messages)

        assert r3.content == "last"
        assert r4.content == "last"

    @pytest.mark.asyncio
    async def test_call_count_property(self) -> None:
        """call_count property should track calls."""
        from tests.mocks.mock_provider import MockProviderWithCounter

        provider = MockProviderWithCounter(responses_by_call=["response"])
        messages = [Message(role="user", content="Hello")]

        assert provider.call_count == 0
        await provider.complete(messages)
        assert provider.call_count == 1
        await provider.complete(messages)
        assert provider.call_count == 2
