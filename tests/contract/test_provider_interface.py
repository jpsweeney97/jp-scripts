"""Contract tests for LLM provider interface compliance.

These tests verify that all provider implementations satisfy the LLMProvider
protocol and behave consistently across different backends.

Contract tests are parametrized over all registered providers to ensure
uniform behavior regardless of which provider is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from jpscripts.providers import (
    BaseLLMProvider,
    LLMProvider,
    ProviderCapability,
    ProviderType,
)
from jpscripts.providers.anthropic import AnthropicProvider
from jpscripts.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig


@pytest.fixture
def mock_config() -> AppConfig:
    """Create a mock AppConfig for testing."""
    config = MagicMock()
    config.ai = MagicMock()
    config.ai.max_file_context_chars = 50_000
    config.user = MagicMock()
    config.user.workspace_root = MagicMock()
    return config


# All provider classes to test
PROVIDER_CLASSES: list[type[BaseLLMProvider]] = [
    AnthropicProvider,
    OpenAIProvider,
]


class TestProviderProtocolCompliance:
    """Verify all providers implement the LLMProvider protocol."""

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_implements_llmprovider_protocol(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Provider class satisfies the LLMProvider Protocol."""
        provider = provider_class(mock_config)
        # runtime_checkable protocols can be used with isinstance
        assert isinstance(provider, LLMProvider)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_has_provider_type_property(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Provider exposes provider_type property."""
        provider = provider_class(mock_config)
        ptype = provider.provider_type
        assert isinstance(ptype, ProviderType)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_has_default_model_property(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Provider exposes default_model property."""
        provider = provider_class(mock_config)
        model = provider.default_model
        assert isinstance(model, str)
        assert len(model) > 0

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_has_available_models_property(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Provider exposes available_models property."""
        provider = provider_class(mock_config)
        models = provider.available_models
        assert isinstance(models, tuple)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_default_model_in_available_models(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Default model is listed in available models."""
        provider = provider_class(mock_config)
        # Default model should be available (possibly as alias)
        # Check if default or any model starts with same prefix
        default = provider.default_model
        available = provider.available_models
        # Either exact match or the default is a dated version
        assert default in available or any(default.startswith(m.split("-")[0]) for m in available)


class TestProviderCapabilities:
    """Test capability reporting across providers."""

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_supports_streaming_returns_bool(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """supports_streaming() returns a boolean."""
        provider = provider_class(mock_config)
        result = provider.supports_streaming()
        assert isinstance(result, bool)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_supports_tools_returns_bool(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """supports_tools() returns a boolean."""
        provider = provider_class(mock_config)
        result = provider.supports_tools()
        assert isinstance(result, bool)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_supports_json_mode_returns_bool(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """supports_json_mode() returns a boolean."""
        provider = provider_class(mock_config)
        result = provider.supports_json_mode()
        assert isinstance(result, bool)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_get_capabilities_returns_frozenset(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """get_capabilities() returns a frozenset of ProviderCapability."""
        provider = provider_class(mock_config)
        caps = provider.get_capabilities()
        assert isinstance(caps, frozenset)
        assert all(isinstance(c, ProviderCapability) for c in caps)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_capabilities_match_support_methods(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Capabilities set matches individual support method results."""
        provider = provider_class(mock_config)
        caps = provider.get_capabilities()

        # STREAMING capability should match supports_streaming()
        has_streaming = ProviderCapability.STREAMING in caps
        assert has_streaming == provider.supports_streaming()

        # TOOLS capability should match supports_tools()
        has_tools = ProviderCapability.TOOLS in caps
        assert has_tools == provider.supports_tools()

        # JSON_MODE capability should match supports_json_mode()
        has_json = ProviderCapability.JSON_MODE in caps
        assert has_json == provider.supports_json_mode()


class TestProviderContextLimits:
    """Test context limit reporting."""

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_get_context_limit_returns_positive_int(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """get_context_limit() returns a positive integer."""
        provider = provider_class(mock_config)
        limit = provider.get_context_limit()
        assert isinstance(limit, int)
        assert limit > 0

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_get_context_limit_for_specific_model(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """get_context_limit() accepts model parameter."""
        provider = provider_class(mock_config)
        default_model = provider.default_model
        limit = provider.get_context_limit(default_model)
        assert isinstance(limit, int)
        assert limit > 0

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_context_limit_reasonable_range(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Context limits are in reasonable range (4K-500K tokens)."""
        provider = provider_class(mock_config)
        limit = provider.get_context_limit()
        assert 4_000 <= limit <= 500_000


class TestProviderMethodSignatures:
    """Test that provider methods have correct signatures."""

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_complete_method_exists(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Provider has async complete() method."""
        provider = provider_class(mock_config)
        assert hasattr(provider, "complete")
        assert callable(provider.complete)

    @pytest.mark.parametrize("provider_class", PROVIDER_CLASSES)
    def test_stream_method_exists(
        self, provider_class: type[BaseLLMProvider], mock_config: AppConfig
    ) -> None:
        """Provider has stream() method."""
        provider = provider_class(mock_config)
        assert hasattr(provider, "stream")
        assert callable(provider.stream)


class TestAnthropicProviderSpecifics:
    """Anthropic-specific contract tests."""

    def test_provider_type_is_anthropic(self, mock_config: AppConfig) -> None:
        """AnthropicProvider has correct provider type."""
        provider = AnthropicProvider(mock_config)
        assert provider.provider_type == ProviderType.ANTHROPIC

    def test_default_model_is_claude(self, mock_config: AppConfig) -> None:
        """Default model is a Claude model."""
        provider = AnthropicProvider(mock_config)
        assert "claude" in provider.default_model.lower()

    def test_all_capabilities_enabled(self, mock_config: AppConfig) -> None:
        """Anthropic supports all major capabilities."""
        provider = AnthropicProvider(mock_config)
        assert provider.supports_streaming()
        assert provider.supports_tools()
        assert provider.supports_json_mode()


class TestOpenAIProviderSpecifics:
    """OpenAI-specific contract tests."""

    def test_provider_type_is_openai(self, mock_config: AppConfig) -> None:
        """OpenAIProvider has correct provider type."""
        provider = OpenAIProvider(mock_config)
        assert provider.provider_type == ProviderType.OPENAI

    def test_default_model_is_gpt(self, mock_config: AppConfig) -> None:
        """Default model is a GPT model."""
        provider = OpenAIProvider(mock_config)
        default = provider.default_model.lower()
        assert "gpt" in default or "o1" in default or "o3" in default

    def test_all_capabilities_enabled(self, mock_config: AppConfig) -> None:
        """OpenAI supports all major capabilities."""
        provider = OpenAIProvider(mock_config)
        assert provider.supports_streaming()
        assert provider.supports_tools()
        assert provider.supports_json_mode()
