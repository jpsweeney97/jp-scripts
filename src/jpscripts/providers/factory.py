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
    PROVIDER_REGISTRY,
    BaseLLMProvider,
    LLMProvider,
    ModelNotFoundError,
    ProviderError,
    ProviderType,
    infer_provider_type,
)

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig


def parse_provider_type(provider_str: str) -> ProviderType:
    """Parse a provider string to ProviderType enum.

    Args:
        provider_str: Provider name ("anthropic", "openai", "codex")

    Returns:
        The corresponding ProviderType

    Raises:
        ValueError: If provider string is not recognized
    """
    ptype_map = {
        "anthropic": ProviderType.ANTHROPIC,
        "openai": ProviderType.OPENAI,
        "codex": ProviderType.CODEX,
    }
    ptype = ptype_map.get(provider_str.lower())
    if ptype is None:
        raise ValueError(f"Unknown provider: {provider_str}")
    return ptype


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


def _ensure_providers_registered() -> None:
    """Ensure provider modules are imported so decorators run.

    The registry is populated when provider modules are imported,
    which triggers the @register_provider decorators. This function
    lazily imports all provider modules on first use.
    """
    if not PROVIDER_REGISTRY:
        from jpscripts.providers import anthropic, codex, openai  # noqa: F401


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
    """Create a provider instance for the given type using the registry."""
    # Check cache first
    if ptype in pconfig._provider_cache:
        return pconfig._provider_cache[ptype]

    # Ensure provider modules are imported so registry is populated
    _ensure_providers_registered()

    # Look up factory in registry
    factory = PROVIDER_REGISTRY.get(ptype)
    if factory is None:
        raise ProviderError(f"Unknown provider type: {ptype}")

    # Build provider-specific kwargs
    kwargs: dict[str, object] = {}
    if ptype == ProviderType.CODEX:
        kwargs["full_auto"] = pconfig.codex_full_auto
        kwargs["web_enabled"] = pconfig.codex_web_enabled

    # Create and cache the provider
    provider = factory(config, **kwargs)
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
    "parse_provider_type",
]
