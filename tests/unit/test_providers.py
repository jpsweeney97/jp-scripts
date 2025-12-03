"""
Unit tests for the providers module.

These tests verify the provider protocol, type conversions,
and error handling without making actual API calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from jpscripts.providers.anthropic import AnthropicProvider
    from jpscripts.providers.openai import OpenAIProvider

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


class TestParseProviderType:
    """Test parse_provider_type function."""

    def test_parse_anthropic(self) -> None:
        from jpscripts.providers.factory import parse_provider_type

        assert parse_provider_type("anthropic") == ProviderType.ANTHROPIC
        assert parse_provider_type("ANTHROPIC") == ProviderType.ANTHROPIC

    def test_parse_openai(self) -> None:
        from jpscripts.providers.factory import parse_provider_type

        assert parse_provider_type("openai") == ProviderType.OPENAI
        assert parse_provider_type("OpenAI") == ProviderType.OPENAI

    def test_parse_unknown(self) -> None:
        from jpscripts.providers.factory import parse_provider_type

        with pytest.raises(ValueError, match="Unknown provider"):
            parse_provider_type("unknown")

    def test_parse_codex_removed(self) -> None:
        from jpscripts.providers.factory import parse_provider_type

        with pytest.raises(ValueError, match="Codex provider has been removed"):
            parse_provider_type("codex")


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
        assert config.fallback_enabled is True
        assert config.web_enabled is False

    def test_config_with_options(self) -> None:
        config = ProviderConfig(
            fallback_enabled=False,
            web_enabled=True,
        )
        assert config.fallback_enabled is False
        assert config.web_enabled is True


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
        system, _converted = _convert_messages_to_anthropic(messages, system_prompt="Be concise")
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
        content_value = converted[0]["content"]
        assert isinstance(content_value, str)
        assert "Be helpful" in content_value

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
        function_payload = converted[0]["function"]
        assert isinstance(function_payload, dict)
        name_value = function_payload.get("name")
        assert isinstance(name_value, str)
        assert name_value == "search"


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


class TestAnthropicErrorHandling:
    """Test Anthropic provider error handling and API key redaction."""

    def test_redact_api_key_from_message(self) -> None:
        from jpscripts.providers.anthropic import _redact_api_key

        # Test Anthropic-style key pattern
        msg = "Error with key sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        redacted = _redact_api_key(msg)
        assert "sk-ant-" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jpscripts.providers.anthropic import _redact_api_key

        fake_key = "sk-ant-test-key-12345678901234567890"
        monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key)
        msg = f"Request failed with key {fake_key}"
        redacted = _redact_api_key(msg)
        assert fake_key not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_generic_secret_patterns(self) -> None:
        from jpscripts.providers.anthropic import _redact_api_key

        msg = "api_key=super_secret_value_12345678"
        redacted = _redact_api_key(msg)
        assert "super_secret" not in redacted

    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jpscripts.core.config import AppConfig
        from jpscripts.providers import AuthenticationError, ProviderError
        from jpscripts.providers.anthropic import AnthropicProvider

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = AnthropicProvider(AppConfig())
        # If anthropic package is not installed, it raises ProviderError first
        # If installed but no key, it raises AuthenticationError
        try:
            import anthropic  # noqa: F401

            with pytest.raises(AuthenticationError, match="ANTHROPIC_API_KEY"):
                provider._get_client()
        except ImportError:
            with pytest.raises(ProviderError, match="anthropic package not installed"):
                provider._get_client()

    def test_missing_package_raises_provider_error(self) -> None:
        # Skip if anthropic is installed - can't easily test missing import
        try:
            import anthropic  # noqa: F401

            pytest.skip("anthropic package is installed")
        except ImportError:
            from jpscripts.core.config import AppConfig
            from jpscripts.providers import ProviderError
            from jpscripts.providers.anthropic import AnthropicProvider

            provider = AnthropicProvider(AppConfig())
            with pytest.raises(ProviderError, match="anthropic package not installed"):
                provider._get_client()


class TestOpenAIErrorHandling:
    """Test OpenAI provider error handling and API key redaction."""

    def test_redact_api_key_from_message(self) -> None:
        from jpscripts.providers.openai import _redact_api_key

        # Test OpenAI-style key pattern (sk- followed by 20+ chars)
        msg = "Error with key sk-abcdefghijklmnopqrstuvwxyz1234"
        redacted = _redact_api_key(msg)
        assert "[REDACTED]" in redacted

    def test_redact_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jpscripts.providers.openai import _redact_api_key

        fake_key = "sk-test-key-1234567890123456789012345"
        monkeypatch.setenv("OPENAI_API_KEY", fake_key)
        msg = f"Request failed with key {fake_key}"
        redacted = _redact_api_key(msg)
        assert fake_key not in redacted
        assert "[REDACTED]" in redacted

    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jpscripts.core.config import AppConfig
        from jpscripts.providers import AuthenticationError, ProviderError
        from jpscripts.providers.openai import OpenAIProvider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(AppConfig())
        # If openai package is not installed, it raises ProviderError first
        # If installed but no key, it raises AuthenticationError
        try:
            import openai  # noqa: F401

            with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
                provider._get_client()
        except ImportError:
            with pytest.raises(ProviderError, match="openai package not installed"):
                provider._get_client()


class TestAnthropicHandleApiError:
    """Test Anthropic provider _handle_api_error method."""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        from jpscripts.core.config import AppConfig
        from jpscripts.providers.anthropic import AnthropicProvider

        return AnthropicProvider(AppConfig())

    def test_handle_authentication_error(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts AuthenticationError correctly."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from jpscripts.providers import AuthenticationError

        exc = anthropic.AuthenticationError("Invalid API key")
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            provider._handle_api_error(exc)

    def test_handle_rate_limit_error(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts RateLimitError correctly."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from jpscripts.providers import RateLimitError

        exc = anthropic.RateLimitError("Rate limit exceeded")
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            provider._handle_api_error(exc)

    def test_handle_not_found_error(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts NotFoundError to ModelNotFoundError."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from jpscripts.providers import ModelNotFoundError

        exc = anthropic.NotFoundError("Model not found")
        with pytest.raises(ModelNotFoundError, match="Model not found"):
            provider._handle_api_error(exc)

    def test_handle_bad_request_context_error(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts context-related BadRequestError to ContextLengthError."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from jpscripts.providers import ContextLengthError

        exc = anthropic.BadRequestError("context length exceeded maximum token limit")
        with pytest.raises(ContextLengthError):
            provider._handle_api_error(exc)

    def test_handle_bad_request_content_filter_error(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts content-related BadRequestError to ContentFilterError."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from jpscripts.providers import ContentFilterError

        exc = anthropic.BadRequestError("content blocked by safety filter")
        with pytest.raises(ContentFilterError):
            provider._handle_api_error(exc)

    def test_handle_bad_request_generic_error(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts generic BadRequestError to ProviderError."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from jpscripts.providers import ProviderError

        exc = anthropic.BadRequestError("invalid request format")
        with pytest.raises(ProviderError, match="invalid request format"):
            provider._handle_api_error(exc)

    def test_handle_generic_api_error(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts generic APIError to ProviderError."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from jpscripts.providers import ProviderError

        exc = anthropic.APIError("server error")
        with pytest.raises(ProviderError, match="server error"):
            provider._handle_api_error(exc)

    def test_handle_unknown_exception(self, provider: AnthropicProvider) -> None:
        """_handle_api_error converts unknown exceptions to ProviderError."""
        from jpscripts.providers import ProviderError

        exc = ValueError("unexpected error")
        with pytest.raises(ProviderError, match="unexpected error"):
            provider._handle_api_error(exc)

    def test_handle_error_redacts_api_key(
        self, provider: AnthropicProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_handle_api_error redacts API keys from error messages."""
        from jpscripts.providers import ProviderError

        fake_key = "sk-ant-api03-secretkey123456789012345678"
        monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key)
        exc = ValueError(f"Error authenticating with key {fake_key}")

        with pytest.raises(ProviderError) as exc_info:
            provider._handle_api_error(exc)

        assert fake_key not in str(exc_info.value)


class TestOpenAIHandleApiError:
    """Test OpenAI provider _handle_api_error method."""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        from jpscripts.core.config import AppConfig
        from jpscripts.providers.openai import OpenAIProvider

        return OpenAIProvider(AppConfig())

    def test_handle_authentication_error(self, provider: OpenAIProvider) -> None:
        """_handle_api_error converts AuthenticationError correctly."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        from jpscripts.providers import AuthenticationError

        exc = openai.AuthenticationError(
            "Invalid API key",
            response=None,
            body=None,  # type: ignore[arg-type]
        )
        with pytest.raises(AuthenticationError):
            provider._handle_api_error(exc)

    def test_handle_rate_limit_error(self, provider: OpenAIProvider) -> None:
        """_handle_api_error converts RateLimitError correctly."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        from jpscripts.providers import RateLimitError

        exc = openai.RateLimitError(
            "Rate limit exceeded",
            response=None,
            body=None,  # type: ignore[arg-type]
        )
        with pytest.raises(RateLimitError):
            provider._handle_api_error(exc)

    def test_handle_not_found_error(self, provider: OpenAIProvider) -> None:
        """_handle_api_error converts NotFoundError to ModelNotFoundError."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        from jpscripts.providers import ModelNotFoundError

        exc = openai.NotFoundError(
            "Model not found",
            response=None,
            body=None,  # type: ignore[arg-type]
        )
        with pytest.raises(ModelNotFoundError):
            provider._handle_api_error(exc)

    def test_handle_bad_request_context_error(self, provider: OpenAIProvider) -> None:
        """_handle_api_error converts context-related BadRequestError to ContextLengthError."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        from jpscripts.providers import ContextLengthError

        exc = openai.BadRequestError(
            "maximum context length exceeded",
            response=None,  # type: ignore[arg-type]
            body=None,
        )
        with pytest.raises(ContextLengthError):
            provider._handle_api_error(exc)

    def test_handle_bad_request_content_filter_error(self, provider: OpenAIProvider) -> None:
        """_handle_api_error converts content-related BadRequestError to ContentFilterError."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        from jpscripts.providers import ContentFilterError

        exc = openai.BadRequestError(
            "content policy violation",
            response=None,  # type: ignore[arg-type]
            body=None,
        )
        with pytest.raises(ContentFilterError):
            provider._handle_api_error(exc)

    def test_handle_unknown_exception(self, provider: OpenAIProvider) -> None:
        """_handle_api_error converts unknown exceptions to ProviderError."""
        from jpscripts.providers import ProviderError

        exc = ValueError("unexpected error")
        with pytest.raises(ProviderError, match="unexpected error"):
            provider._handle_api_error(exc)

    def test_handle_error_redacts_api_key(
        self, provider: OpenAIProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_handle_api_error redacts API keys from error messages."""
        from jpscripts.providers import ProviderError

        fake_key = "sk-proj-secretkey12345678901234567890"
        monkeypatch.setenv("OPENAI_API_KEY", fake_key)
        exc = ValueError(f"Error authenticating with key {fake_key}")

        with pytest.raises(ProviderError) as exc_info:
            provider._handle_api_error(exc)

        assert fake_key not in str(exc_info.value)
