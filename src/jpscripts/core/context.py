"""Context gathering and token management re-exports.

This module provides a unified interface for context-related utilities:
    - Context gathering from commands and files
    - Token counting and budget management
    - File reading with skeleton extraction
"""

from __future__ import annotations

from jpscripts.ai.tokens import TRUNCATION_MARKER, TokenBudgetManager, TokenCounter
from jpscripts.core.context_gatherer import (
    DEFAULT_MODEL_CONTEXT_LIMIT,
    FILE_PATTERN,
    SENSITIVE_PATTERNS,
    STRUCTURED_EXTENSIONS,
    SYNTAX_WARNING,
    GatherContextResult,
    gather_context,
    get_file_skeleton,
    read_file_context,
    resolve_files_from_output,
    run_and_capture,
    smart_read_context,
)

__all__ = [
    "DEFAULT_MODEL_CONTEXT_LIMIT",
    "FILE_PATTERN",
    "SENSITIVE_PATTERNS",
    "STRUCTURED_EXTENSIONS",
    "SYNTAX_WARNING",
    "TRUNCATION_MARKER",
    "GatherContextResult",
    "TokenBudgetManager",
    "TokenCounter",
    "gather_context",
    "get_file_skeleton",
    "read_file_context",
    "resolve_files_from_output",
    "run_and_capture",
    "smart_read_context",
]
