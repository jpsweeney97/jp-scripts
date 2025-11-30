"""
Provider factory for creating LLM provider instances.

This module provides a unified way to create provider instances based on
configuration, model ID, or explicit provider type selection.

Usage:
    from jpscripts.providers.factory import get_provider, ProviderConfig

    # Auto-detect provider from model ID
    provider = get_provider(config, model_id="claude-opus-4-5")

    # Explicit provider selection
    provider = get_provider(config, provider_type=ProviderType.ANTHROPIC)

    # With Codex fallback preference
    provider = get_provider(
        config,
        model_id="o1",
        prefer_codex=True,  # Use Codex CLI if available
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from jpscripts.providers import (
    BaseLLMProvider,
    LLMProvider,
    ModelNotFoundError,
    ProviderError,
    ProviderType,
    infer_provider_type,
)

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig


@dataclass
class ProviderConfig:
    """Configuration for provider instantiation.

    Attributes:
        prefer_codex: If True, prefer Codex CLI for OpenAI models when available
        codex_full_auto: Run Codex in full-auto mode
        codex_web_enabled: Enable web search for Codex
        fallback_enabled: If True, fall back to other providers on failure
    """

    prefer_codex: bool = False
    codex_full_auto: bool = False
    codex_web_enabled: bool = False
    fallback_enabled: bool = True
    _provider_cache: dict[ProviderType, BaseLLMProvider] = field(default_factory=dict, repr=False)


def _create_anthropic_provider(config: AppConfig) -> BaseLLMProvider:
    """Create an Anthropic provider instance."""
    from jpscripts.providers.anthropic import AnthropicProvider

    return AnthropicProvider(config)


def _create_openai_provider(config: AppConfig) -> BaseLLMProvider:
    """Create an OpenAI provider instance."""
    from jpscripts.providers.openai import OpenAIProvider

    return OpenAIProvider(config)


def _create_codex_provider(
    config: AppConfig,
    full_auto: bool = False,
    web_enabled: bool = False,
) -> BaseLLMProvider:
    """Create a Codex provider instance."""
    from jpscripts.providers.codex import CodexProvider

    return CodexProvider(config, full_auto=full_auto, web_enabled=web_enabled)


def get_provider(
    config: AppConfig,
    *,
    model_id: str | None = None,
    provider_type: ProviderType | None = None,
    provider_config: ProviderConfig | None = None,
) -> LLMProvider:
    """Get an LLM provider instance.

    This is the main entry point for obtaining provider instances.
    It handles provider selection, instantiation, and caching.

    Args:
        config: Application configuration
        model_id: Model ID to use (determines provider if not specified)
        provider_type: Explicit provider type (overrides model-based inference)
        provider_config: Additional provider configuration

    Returns:
        An LLMProvider instance

    Raises:
        ModelNotFoundError: If provider cannot be determined
        ProviderError: If provider instantiation fails

    Examples:
        # Auto-detect from model
        provider = get_provider(config, model_id="claude-opus-4-5")

        # Explicit provider
        provider = get_provider(config, provider_type=ProviderType.OPENAI)

        # With Codex preference
        pconfig = ProviderConfig(prefer_codex=True, codex_full_auto=True)
        provider = get_provider(config, model_id="o1", provider_config=pconfig)
    """
    pconfig = provider_config or ProviderConfig()

    # Determine provider type
    ptype: ProviderType
    if provider_type is not None:
        ptype = provider_type
    elif model_id is not None:
        ptype = infer_provider_type(model_id)
    else:
        # Default to config's default model
        default_model = getattr(config, "default_model", "claude-sonnet-4-5")
        ptype = infer_provider_type(default_model)

    # Check for Codex preference for OpenAI models
    if ptype == ProviderType.OPENAI and pconfig.prefer_codex:
        from jpscripts.providers.codex import is_codex_available

        if is_codex_available():
            ptype = ProviderType.CODEX

    # Create provider instance
    return _instantiate_provider(config, ptype, pconfig)


def _instantiate_provider(
    config: AppConfig,
    ptype: ProviderType,
    pconfig: ProviderConfig,
) -> BaseLLMProvider:
    """Create a provider instance for the given type."""
    # Check cache first
    if ptype in pconfig._provider_cache:
        return pconfig._provider_cache[ptype]

    provider: BaseLLMProvider

    if ptype == ProviderType.ANTHROPIC:
        provider = _create_anthropic_provider(config)
    elif ptype == ProviderType.OPENAI:
        provider = _create_openai_provider(config)
    elif ptype == ProviderType.CODEX:
        provider = _create_codex_provider(
            config,
            full_auto=pconfig.codex_full_auto,
            web_enabled=pconfig.codex_web_enabled,
        )
    else:
        raise ProviderError(f"Unknown provider type: {ptype}")

    # Cache the provider
    pconfig._provider_cache[ptype] = provider
    return provider


def get_provider_for_model(config: AppConfig, model_id: str) -> LLMProvider:
    """Convenience function to get a provider for a specific model.

    This is a simpler alternative to get_provider() when you just
    need a provider for a known model ID.

    Args:
        config: Application configuration
        model_id: The model ID to use

    Returns:
        An appropriate LLMProvider for the model
    """
    return get_provider(config, model_id=model_id)


@lru_cache(maxsize=1)
def get_default_provider(config: AppConfig) -> LLMProvider:
    """Get the default provider based on configuration.

    This function caches the result for repeated calls with the same config.

    Args:
        config: Application configuration

    Returns:
        The default LLMProvider
    """
    return get_provider(config)


def list_available_models() -> dict[ProviderType, tuple[str, ...]]:
    """List all available models by provider.

    Returns:
        Dict mapping provider type to tuple of model IDs
    """
    from jpscripts.providers.anthropic import ANTHROPIC_AVAILABLE_MODELS
    from jpscripts.providers.codex import CODEX_AVAILABLE_MODELS
    from jpscripts.providers.openai import OPENAI_AVAILABLE_MODELS

    return {
        ProviderType.ANTHROPIC: ANTHROPIC_AVAILABLE_MODELS,
        ProviderType.OPENAI: OPENAI_AVAILABLE_MODELS,
        ProviderType.CODEX: CODEX_AVAILABLE_MODELS,
    }


def get_model_context_limit(model_id: str) -> int:
    """Get the context limit for a model.

    Args:
        model_id: The model ID

    Returns:
        Context limit in tokens

    Raises:
        ModelNotFoundError: If model is not recognized
    """
    from jpscripts.providers.anthropic import ANTHROPIC_CONTEXT_LIMITS
    from jpscripts.providers.codex import CODEX_CONTEXT_LIMITS
    from jpscripts.providers.openai import OPENAI_CONTEXT_LIMITS

    # Check all provider limits
    for limits in [ANTHROPIC_CONTEXT_LIMITS, OPENAI_CONTEXT_LIMITS, CODEX_CONTEXT_LIMITS]:
        if model_id in limits:
            return limits[model_id]

    # Try to infer and get from provider
    try:
        ptype = infer_provider_type(model_id)
        if ptype == ProviderType.ANTHROPIC:
            return 200_000  # Default for Claude
        if ptype in (ProviderType.OPENAI, ProviderType.CODEX):
            return 128_000  # Default for GPT
    except ModelNotFoundError:
        pass

    raise ModelNotFoundError(f"Unknown model: {model_id}")


__all__ = [
    "ProviderConfig",
    "get_default_provider",
    "get_model_context_limit",
    "get_provider",
    "get_provider_for_model",
    "list_available_models",
]
