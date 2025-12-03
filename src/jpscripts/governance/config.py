"""Safety rules configuration loading for governance checks.

Loads declarative safety rules from safety_rules.yaml and provides
typed access to rule definitions. Falls back to embedded defaults
if the YAML file is missing or invalid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class SecretPrefix:
    """Known API key prefix pattern."""

    prefix: str
    description: str
    min_length: int = 16


@dataclass(frozen=True)
class SafetyConfig:
    """Loaded safety rules configuration.

    All sets are frozen for immutability and hashability.
    """

    # Subprocess functions that block when called in async context
    blocking_subprocess_funcs: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "run",
                "call",
                "check_call",
                "check_output",
                "Popen",
                "getoutput",
                "getstatusoutput",
            }
        )
    )

    # Destructive filesystem operations
    destructive_shutil_funcs: frozenset[str] = field(default_factory=lambda: frozenset({"rmtree"}))

    destructive_os_funcs: frozenset[str] = field(
        default_factory=lambda: frozenset({"remove", "unlink"})
    )

    # Dynamic execution builtins (no safety override allowed)
    forbidden_dynamic_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset({"eval", "exec", "compile", "__import__"})
    )

    # Debug modules and functions
    debug_modules: frozenset[str] = field(default_factory=lambda: frozenset({"pdb", "ipdb"}))

    debug_builtins: frozenset[str] = field(default_factory=lambda: frozenset({"breakpoint"}))

    # Exit builtins
    exit_builtins: frozenset[str] = field(default_factory=lambda: frozenset({"quit", "exit"}))

    # Secret detection - sensitive variable name keywords
    secret_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"}
        )
    )

    # Known API key prefixes
    known_secret_prefixes: tuple[SecretPrefix, ...] = field(
        default_factory=lambda: (
            SecretPrefix("sk-", "OpenAI API key", 20),
            SecretPrefix("gsk_", "Groq API key", 20),
            SecretPrefix("pk_", "Stripe public key", 20),
            SecretPrefix("sk_", "Stripe secret key", 20),
            SecretPrefix("xoxb-", "Slack bot token", 20),
            SecretPrefix("xoxp-", "Slack user token", 20),
            SecretPrefix("ghp_", "GitHub PAT", 20),
            SecretPrefix("gho_", "GitHub OAuth token", 20),
            SecretPrefix("AIza", "Google API key", 20),
            SecretPrefix("AKIA", "AWS access key", 16),
        )
    )

    # Minimum entropy for secret detection
    secret_min_entropy: float = 4.0

    # Safety override comment pattern
    safety_override_pattern: str = "# safety: checked"

    # Config version
    version: str = "1.0"


def _get_templates_dir() -> Path:
    """Get the templates directory path."""
    return Path(__file__).parent.parent / "templates"


def _load_yaml_config() -> dict[str, Any] | None:
    """Load and parse the YAML config file.

    Returns None if file doesn't exist or is invalid.
    """
    try:
        import yaml  # Lazy import to avoid startup cost
    except ImportError:
        return None

    yaml_path = _get_templates_dir() / "safety_rules.yaml"
    if not yaml_path.exists():
        return None

    try:
        with yaml_path.open(encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
            return data
    except (OSError, yaml.YAMLError):
        return None


def _extract_funcs_from_yaml(
    yaml_data: Mapping[str, Any], section: str, module: str, key: str = "functions"
) -> frozenset[str]:
    """Extract function names from YAML config section."""
    result: set[str] = set()

    forbidden_calls = yaml_data.get("forbidden_calls", [])
    for rule in forbidden_calls:
        if rule.get("type") == section:
            # Direct module/functions format
            if rule.get("module") == module and key in rule:
                result.update(rule[key])

            # Nested rules format
            for sub_rule in rule.get("rules", []):
                if sub_rule.get("module") == module and key in sub_rule:
                    result.update(sub_rule[key])

    return frozenset(result)


def _extract_builtins_from_yaml(yaml_data: Mapping[str, Any], section: str) -> frozenset[str]:
    """Extract builtin names from YAML config section."""
    result: set[str] = set()

    forbidden_calls = yaml_data.get("forbidden_calls", [])
    for rule in forbidden_calls:
        if rule.get("type") == section:
            # Direct builtins format
            if "builtins" in rule:
                result.update(rule["builtins"])

            # Nested rules format
            for sub_rule in rule.get("rules", []):
                if "builtins" in sub_rule:
                    result.update(sub_rule["builtins"])

    return frozenset(result)


def _extract_secret_prefixes(yaml_data: Mapping[str, Any]) -> tuple[SecretPrefix, ...]:
    """Extract known secret prefixes from YAML config."""
    result: list[SecretPrefix] = []

    secret_detection = yaml_data.get("secret_detection", [])
    for rule in secret_detection:
        for prefix_def in rule.get("known_prefixes", []):
            result.append(
                SecretPrefix(
                    prefix=prefix_def["prefix"],
                    description=prefix_def.get("description", ""),
                    min_length=prefix_def.get("min_length", 16),
                )
            )

    return tuple(result) if result else SafetyConfig().known_secret_prefixes


@lru_cache(maxsize=1)
def load_safety_config() -> SafetyConfig:
    """Load safety configuration from YAML or return defaults.

    Uses LRU cache to avoid repeated file I/O.
    """
    yaml_data = _load_yaml_config()

    if yaml_data is None:
        return SafetyConfig()

    # Extract configurable values from YAML, falling back to defaults
    defaults = SafetyConfig()

    blocking_funcs = _extract_funcs_from_yaml(yaml_data, "SYNC_SUBPROCESS", "subprocess")
    destructive_shutil = _extract_funcs_from_yaml(yaml_data, "DESTRUCTIVE_FS", "shutil")
    destructive_os = _extract_funcs_from_yaml(yaml_data, "DESTRUCTIVE_FS", "os")
    forbidden_dynamic = _extract_builtins_from_yaml(yaml_data, "DYNAMIC_EXECUTION")
    debug_builtins = _extract_builtins_from_yaml(yaml_data, "DEBUG_LEFTOVER")
    exit_builtins = _extract_builtins_from_yaml(yaml_data, "PROCESS_EXIT")
    secret_prefixes = _extract_secret_prefixes(yaml_data)

    return SafetyConfig(
        blocking_subprocess_funcs=blocking_funcs or defaults.blocking_subprocess_funcs,
        destructive_shutil_funcs=destructive_shutil or defaults.destructive_shutil_funcs,
        destructive_os_funcs=destructive_os or defaults.destructive_os_funcs,
        forbidden_dynamic_builtins=forbidden_dynamic or defaults.forbidden_dynamic_builtins,
        debug_builtins=debug_builtins or defaults.debug_builtins,
        exit_builtins=exit_builtins or defaults.exit_builtins,
        known_secret_prefixes=secret_prefixes,
        version=yaml_data.get("version", "1.0"),
    )


def build_known_api_key_pattern(config: SafetyConfig | None = None) -> re.Pattern[str]:
    """Build regex pattern for known API key prefixes from config."""
    if config is None:
        config = load_safety_config()

    # Build alternation pattern from known prefixes
    prefix_patterns = []
    for p in config.known_secret_prefixes:
        # Escape special regex chars in prefix
        escaped = re.escape(p.prefix)
        # Build pattern: prefix + alphanumeric/dash chars of min_length
        prefix_patterns.append(f"{escaped}[A-Za-z0-9_\\-]{{{p.min_length},}}")

    if not prefix_patterns:
        # Fallback to match nothing if no prefixes configured
        return re.compile(r"(?!)")

    pattern = f"""(?x)
    (?P<quote>['\"])
    (?P<value>
        {"|".join(prefix_patterns)}
    )
    (?P=quote)
    """
    return re.compile(pattern)


__all__ = [
    "SafetyConfig",
    "SecretPrefix",
    "build_known_api_key_pattern",
    "load_safety_config",
]
