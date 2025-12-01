"""Secret detection for constitutional compliance checking."""

from __future__ import annotations

import math
import re
from pathlib import Path

from jpscripts.core.governance.types import Violation, ViolationType

# Pattern for variable assignments: API_KEY = "value"
_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (?P<name>[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_]*)
    \s*=\s*
    (?P<quote>['"]?)(?P<value>[A-Za-z0-9+/=_\-]{16,})(?P=quote)
    """
)

# Pattern for dict-style assignments: config["api_key"] = "value" or {"api_key": "value"}
_SECRET_DICT_PATTERN = re.compile(
    r"""(?ix)
    (?P<context>\[|:\s*)
    (?P<quote1>['"])
    (?P<name>[a-z0-9_]*(key|token|secret|password|credential|auth)[a-z0-9_]*)
    (?P=quote1)
    \]\s*=\s*|\s*:\s*  # Either dict assignment or dict literal
    (?P<quote2>['"])(?P<value>[A-Za-z0-9+/=_\-]{16,})(?P=quote2)
    """
)

# Pattern for known API key prefixes (sk-, gsk_, pk_, xoxb-, etc.)
_KNOWN_API_KEY_PATTERN = re.compile(
    r"""(?x)
    (?P<quote>['"])
    (?P<value>
        sk-[A-Za-z0-9]{20,}|           # OpenAI
        gsk_[A-Za-z0-9]{20,}|          # Groq
        pk_[A-Za-z0-9]{20,}|           # Stripe public key
        sk_[A-Za-z0-9]{20,}|           # Stripe secret key
        xoxb-[A-Za-z0-9\-]{20,}|       # Slack bot token
        xoxp-[A-Za-z0-9\-]{20,}|       # Slack user token
        ghp_[A-Za-z0-9]{20,}|          # GitHub PAT
        gho_[A-Za-z0-9]{20,}|          # GitHub OAuth
        AIza[A-Za-z0-9_\-]{20,}|       # Google API key
        AKIA[A-Z0-9]{16}               # AWS access key
    )
    (?P=quote)
    """
)


def check_for_secrets(content: str, file_path: Path) -> list[Violation]:
    """Detect obvious secret-like assignments in content.

    Uses multiple detection strategies:
    1. Variable assignments (API_KEY = "value")
    2. Dict-style assignments (config["api_key"] = "value")
    3. Known API key prefixes (sk-, ghp_, etc.)
    """
    lines = content.splitlines()
    violations: list[Violation] = []
    seen_positions: set[int] = set()  # Avoid duplicate violations

    def _add_violation(match: re.Match[str], name: str, value: str) -> None:
        """Helper to add a violation if not already detected at this position."""
        pos = match.start()
        if pos in seen_positions:
            return
        seen_positions.add(pos)

        line = content.count("\n", 0, pos) + 1
        column = pos - content.rfind("\n", 0, pos) - 1

        # Check for safety override comment on the same line
        if 0 < line <= len(lines) and "# safety: checked" in lines[line - 1]:
            return

        violations.append(
            Violation(
                type=ViolationType.SECRET_LEAK,
                file=file_path,
                line=line,
                column=column,
                message=f"Potential secret detected in {name} with high-entropy value.",
                suggestion="Remove the secret, rotate credentials, and load from environment or secret manager.",
                severity="error",
                fatal=True,
            )
        )

    # Strategy 1: Variable assignments (API_KEY = "value")
    for match in _SECRET_PATTERN.finditer(content):
        name = match.group("name")
        value = match.group("value")
        entropy = _estimate_entropy(value)
        # Threshold of 4.0 bits better catches real API keys while avoiding false positives
        if entropy >= 4.0:
            _add_violation(match, name, value)

    # Strategy 2: Dict-style assignments (config["api_key"] = "value")
    for match in _SECRET_DICT_PATTERN.finditer(content):
        name = match.group("name")
        value = match.group("value")
        entropy = _estimate_entropy(value)
        if entropy >= 4.0:
            _add_violation(match, name, value)

    # Strategy 3: Known API key prefixes - no entropy check needed
    for match in _KNOWN_API_KEY_PATTERN.finditer(content):
        value = match.group("value")
        # Determine type from prefix
        prefix = value.split("-")[0] if "-" in value else value[:4]
        name = f"API key ({prefix})"
        _add_violation(match, name, value)

    return violations


def _estimate_entropy(value: str) -> float:
    """Rough entropy estimator for secret-like strings."""
    if not value:
        return 0.0
    freq = {ch: value.count(ch) for ch in set(value)}
    length = len(value)

    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log(p, 2)
    return entropy


__all__ = [
    "check_for_secrets",
]
