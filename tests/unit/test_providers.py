"""
Unit tests for the providers module.

These tests verify the provider protocol, type conversions,
and error handling without making actual API calls.
"""

from __future__ import annotations

import pytest

from jpscripts.providers import (
    CompletionOptions,
    CompletionResponse,
    Message,
    ModelNotFoundError,
    ProviderError,
    ProviderType,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    infer_provider_type,
)
from jpscripts.providers.factory import (
    ProviderConfig,
    get_model_context_limit,
    list_available_models,
)


class TestProviderTypes:
    """Test provider type enums and inference."""

    def test_provider_type_values(self) -> None:
        assert ProviderType.ANTHROPIC.value == 1
        assert ProviderType.OPENAI.value == 2
        assert ProviderType.CODEX.value == 3

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("claude-opus-4-5", ProviderType.ANTHROPIC),
            ("claude-sonnet-4-5", ProviderType.ANTHROPIC),
            ("claude-3-opus-20240229", ProviderType.ANTHROPIC),
            ("gpt-4o", ProviderType.OPENAI),
            ("gpt-4-turbo", ProviderType.OPENAI),
            ("o1", ProviderType.OPENAI),
            ("o1-mini", ProviderType.OPENAI),
        ],
    )
    def test_infer_provider_type(self, model_id: str, expected: ProviderType) -> None:
        assert infer_provider_type(model_id) == expected

    def test_infer_provider_type_unknown(self) -> None:
        with pytest.raises(ModelNotFoundError):
            infer_provider_type("unknown-model-xyz")


class TestMessageType:
    """Test Message dataclass."""

    def test_message_basic(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_with_name(self) -> None:
        msg = Message(role="assistant", content="Hi", name="Claude")
        assert msg.name == "Claude"

    def test_message_immutable(self) -> None:
        msg = Message(role="user", content="Test")
        with pytest.raises(AttributeError):
            msg.content = "Changed"  # type: ignore[misc]


class TestTokenUsage:
    """Test TokenUsage dataclass."""

    def test_token_usage_auto_total(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_token_usage_explicit_total(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=160)
        assert usage.total_tokens == 160

    def test_token_usage_immutable(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        with pytest.raises(AttributeError):
            usage.prompt_tokens = 200  # type: ignore[misc]


class TestCompletionResponse:
    """Test CompletionResponse dataclass."""

    def test_completion_response_basic(self) -> None:
        response = CompletionResponse(content="Hello!", model="test-model")
        assert response.content == "Hello!"
        assert response.model == "test-model"
        assert response.finish_reason is None
        assert response.tool_calls == []
        assert response.usage is None

    def test_completion_response_with_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        response = CompletionResponse(
            content="Response",
            model="test",
            usage=usage,
        )
        assert response.usage is not None
        assert response.usage.total_tokens == 15

    def test_completion_response_with_tool_calls(self) -> None:
        tool_call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
        )
        response = CompletionResponse(
            content="",
            model="test",
            tool_calls=[tool_call],
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"


class TestStreamChunk:
    """Test StreamChunk dataclass."""

    def test_stream_chunk_basic(self) -> None:
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.finish_reason is None

    def test_stream_chunk_with_finish(self) -> None:
        chunk = StreamChunk(content="", finish_reason="stop")
        assert chunk.finish_reason == "stop"


class TestCompletionOptions:
    """Test CompletionOptions dataclass."""

    def test_options_defaults(self) -> None:
        opts = CompletionOptions()
        assert opts.temperature is None
        assert opts.max_tokens is None
        assert opts.json_mode is False
        assert opts.tools is None

    def test_options_with_tools(self) -> None:
        tool = ToolDefinition(
            name="search",
            description="Search for info",
            parameters={"type": "object", "properties": {}},
        )
        opts = CompletionOptions(tools=(tool,))
        assert opts.tools is not None
        assert len(opts.tools) == 1


class TestProviderFactory:
    """Test provider factory functions."""

    def test_list_available_models(self) -> None:
        models = list_available_models()
        assert ProviderType.ANTHROPIC in models
        assert ProviderType.OPENAI in models
        assert ProviderType.CODEX in models
        assert "claude-opus-4-5-20251101" in models[ProviderType.ANTHROPIC]
        assert "gpt-4o-2024-11-20" in models[ProviderType.OPENAI]

    @pytest.mark.parametrize(
        "model_id,expected_limit",
        [
            ("claude-opus-4-5", 200_000),
            ("claude-3-haiku", 200_000),
            ("gpt-4o", 128_000),
            ("o1", 200_000),
        ],
    )
    def test_get_model_context_limit(self, model_id: str, expected_limit: int) -> None:
        assert get_model_context_limit(model_id) == expected_limit

    def test_get_model_context_limit_unknown(self) -> None:
        with pytest.raises(ModelNotFoundError):
            get_model_context_limit("unknown-model-xyz")


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_config_defaults(self) -> None:
        config = ProviderConfig()
        assert config.prefer_codex is False
        assert config.codex_full_auto is False
        assert config.codex_web_enabled is False
        assert config.fallback_enabled is True

    def test_config_with_options(self) -> None:
        config = ProviderConfig(
            prefer_codex=True,
            codex_full_auto=True,
        )
        assert config.prefer_codex is True
        assert config.codex_full_auto is True


class TestAnthropicProvider:
    """Test Anthropic provider utilities."""

    def test_model_aliases(self) -> None:
        from jpscripts.providers.anthropic import _resolve_model_id

        assert _resolve_model_id("claude-opus-4-5") == "claude-opus-4-5-20251101"
        assert _resolve_model_id("claude-sonnet-4") == "claude-sonnet-4-20250514"
        assert _resolve_model_id("claude-3-haiku-20240307") == "claude-3-haiku-20240307"

    def test_message_conversion(self) -> None:
        from jpscripts.providers.anthropic import _convert_messages_to_anthropic

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        system, converted = _convert_messages_to_anthropic(messages)
        assert system == "You are helpful"
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_message_conversion_with_system_prompt(self) -> None:
        from jpscripts.providers.anthropic import _convert_messages_to_anthropic

        messages = [Message(role="user", content="Hello")]
        system, _converted = _convert_messages_to_anthropic(
            messages, system_prompt="Be concise"
        )
        assert system == "Be concise"

    def test_tool_conversion(self) -> None:
        from jpscripts.providers.anthropic import _convert_tools_to_anthropic

        tools = (
            ToolDefinition(
                name="search",
                description="Search for info",
                parameters={"type": "object"},
            ),
        )
        converted = _convert_tools_to_anthropic(tools)
        assert converted is not None
        assert len(converted) == 1
        assert converted[0]["name"] == "search"
        assert "input_schema" in converted[0]


class TestOpenAIProvider:
    """Test OpenAI provider utilities."""

    def test_model_aliases(self) -> None:
        from jpscripts.providers.openai import _resolve_model_id

        assert _resolve_model_id("gpt-4o") == "gpt-4o-2024-11-20"
        assert _resolve_model_id("o1") == "o1-2024-12-17"
        assert _resolve_model_id("gpt-4-turbo") == "gpt-4-turbo"

    def test_message_conversion_standard(self) -> None:
        from jpscripts.providers.openai import _convert_messages_to_openai

        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        converted = _convert_messages_to_openai(messages, model="gpt-4o")
        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[1]["role"] == "user"

    def test_message_conversion_o1_no_system(self) -> None:
        from jpscripts.providers.openai import _convert_messages_to_openai

        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        converted = _convert_messages_to_openai(messages, model="o1")
        # o1 doesn't support system messages, so it should be merged
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert "Be helpful" in converted[0]["content"]

    def test_tool_conversion(self) -> None:
        from jpscripts.providers.openai import _convert_tools_to_openai

        tools = (
            ToolDefinition(
                name="search",
                description="Search for info",
                parameters={"type": "object"},
            ),
        )
        converted = _convert_tools_to_openai(tools)
        assert converted is not None
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "search"


class TestCodexProvider:
    """Test Codex provider utilities."""

    def test_codex_availability(self) -> None:
        from jpscripts.providers.codex import is_codex_available

        # Just test that the function runs without error
        result = is_codex_available()
        assert isinstance(result, bool)

    def test_format_messages(self) -> None:
        from jpscripts.providers.codex import _format_messages_for_codex

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        formatted = _format_messages_for_codex(messages)
        assert "[User]" in formatted
        assert "[Assistant]" in formatted
        assert "Hello" in formatted
        assert "Hi" in formatted

    def test_format_messages_with_system(self) -> None:
        from jpscripts.providers.codex import _format_messages_for_codex

        messages = [Message(role="user", content="Hello")]
        formatted = _format_messages_for_codex(messages, system_prompt="Be helpful")
        assert "[System]" in formatted
        assert "Be helpful" in formatted


class TestProviderErrors:
    """Test provider error types."""

    def test_provider_error(self) -> None:
        from jpscripts.providers import ProviderError

        err = ProviderError("Something went wrong")
        assert str(err) == "Something went wrong"

    def test_authentication_error(self) -> None:
        from jpscripts.providers import AuthenticationError

        err = AuthenticationError("Invalid API key")
        assert isinstance(err, ProviderError)

    def test_rate_limit_error(self) -> None:
        from jpscripts.providers import RateLimitError

        err = RateLimitError("Too many requests", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_context_length_error(self) -> None:
        from jpscripts.providers import ContextLengthError

        err = ContextLengthError("Input too long")
        assert isinstance(err, ProviderError)
