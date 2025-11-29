"""Token budget management with AST-aware slicing support.

This module provides:
- TokenCounter: Token counting backed by tiktoken with fallback
- TokenBudgetManager: Priority-based token budget allocation
- Integration with DependencyWalker for semantic slicing

[invariant:typing] All types are explicit; mypy --strict compliant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Sequence, cast

from jpscripts.core.console import get_logger

if TYPE_CHECKING:
    from jpscripts.core.dependency_walker import DependencyWalker

logger = get_logger(__name__)

DEFAULT_MODEL_CONTEXT_LIMIT = 200_000
TRUNCATION_MARKER = "[...truncated]"


class _EncoderProtocol(Protocol):
    def encode(self, text: str, *, disallowed_special: Sequence[str] | set[str] | tuple[str, ...] = ()) -> list[int]:
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        ...


class TruncationStrategy(Protocol):
    def __call__(self, path: Path, max_chars: int, max_tokens: int | None = None, *, limit: int) -> str:
        ...


class TokenCounter:
    """Token counter backed by tiktoken with heuristic fallback."""

    def __init__(self, default_model: str = "gpt-4o") -> None:
        self.default_model = default_model
        self._encoders: dict[str, _EncoderProtocol | None] = {}
        self._warned_missing = False

    def count_tokens(self, text: str, model: str | None = None) -> int:
        target_model = model or self.default_model
        encoder = self._get_encoder(target_model)
        if encoder is None:
            return self._heuristic_tokens(text)
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception as exc:
            logger.warning("Token counting failed for model %s: %s", target_model, exc)
            return self._heuristic_tokens(text)

    def trim_to_fit(self, text: str, max_tokens: int, model: str | None = None) -> str:
        """Trim text to fit within max_tokens."""
        if max_tokens <= 0:
            return ""

        target_model = model or self.default_model
        encoder = self._get_encoder(target_model)
        if encoder is None:
            return text[: self.tokens_to_characters(max_tokens)]

        try:
            encoded = encoder.encode(text, disallowed_special=())
            if len(encoded) <= max_tokens:
                return text
            return encoder.decode(encoded[:max_tokens])
        except Exception as exc:
            logger.warning("Token trim failed for model %s: %s", target_model, exc)
            return text[: self.tokens_to_characters(max_tokens)]

    def tokens_to_characters(self, tokens: int) -> int:
        """Coarse conversion from tokens to characters (upper bound)."""
        if tokens <= 0:
            return 0
        return tokens * 4

    def _heuristic_tokens(self, text: str) -> int:
        return max(0, len(text) // 4)

    def _get_encoder(self, model: str) -> _EncoderProtocol | None:
        if model in self._encoders:
            return self._encoders[model]

        try:
            import importlib

            tiktoken_module = importlib.import_module("tiktoken")
        except ImportError:
            if not self._warned_missing:
                logger.warning("tiktoken is not installed; falling back to heuristic token estimates.")
                self._warned_missing = True
            self._encoders[model] = None
            return None
        except Exception as exc:  # pragma: no cover - defensive import guard
            logger.warning("Failed to import tiktoken: %s", exc)
            self._encoders[model] = None
            return None

        try:
            encoding_for_model = getattr(tiktoken_module, "encoding_for_model", None)
            if not callable(encoding_for_model):
                raise AttributeError("encoding_for_model is unavailable on tiktoken module")
            encoder = encoding_for_model(model)
        except Exception as exc:
            logger.warning("Failed to load encoding for model %s: %s", model, exc)
            self._encoders[model] = None
            return None

        cached = cast(_EncoderProtocol, encoder)
        self._encoders[model] = cached
        return cached


Priority = Literal[1, 2, 3]


@dataclass
class TokenBudgetManager:
    """Priority-based token budget allocation using precise token counts.

    The manager is pure logic; any I/O-based truncation must be provided via `truncator`.
    """

    total_budget: int
    reserved_budget: int = 0
    model_context_limit: int = DEFAULT_MODEL_CONTEXT_LIMIT
    model: str = "gpt-4o"
    token_counter: TokenCounter = field(default_factory=TokenCounter)
    truncator: TruncationStrategy | None = None
    _used_tokens: int = field(default=0, repr=False)
    _allocations: dict[Priority, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.total_budget < 0:
            raise ValueError("total_budget must be non-negative")
        if self.reserved_budget < 0:
            raise ValueError("reserved_budget must be non-negative")
        if self.reserved_budget > self.total_budget:
            raise ValueError("reserved_budget cannot exceed total_budget")
        if self.model_context_limit <= 0:
            raise ValueError("model_context_limit must be positive")
        self._allocations = {1: 0, 2: 0, 3: 0}

    def remaining(self) -> int:
        """Return remaining token budget available for allocation."""
        return max(0, self.total_budget - self.reserved_budget - self._used_tokens)

    def tokens_to_characters(self, tokens: int) -> int:
        """Convert token budget to a conservative character budget."""
        char_budget = self.token_counter.tokens_to_characters(tokens)
        return min(char_budget, self.token_counter.tokens_to_characters(self.model_context_limit))

    def allocate(
        self,
        priority: Priority,
        content: str,
        source_path: Path | None = None,
    ) -> str:
        """Allocate content within token budget, with optional syntax-aware truncation."""
        if not content:
            return ""

        token_budget = self.remaining()
        if token_budget <= 0:
            return ""

        token_count = self.token_counter.count_tokens(content, model=self.model)
        if token_count <= token_budget:
            self._track_allocation(priority, token_count)
            return content

        truncated = self._truncate_content(content, token_budget, source_path)
        if not truncated:
            return ""

        final_tokens = self.token_counter.count_tokens(truncated, model=self.model)
        if final_tokens > token_budget:
            truncated = self.token_counter.trim_to_fit(truncated, token_budget, model=self.model)
            final_tokens = self.token_counter.count_tokens(truncated, model=self.model)

        self._track_allocation(priority, final_tokens)
        return truncated

    def _track_allocation(self, priority: Priority, tokens: int) -> None:
        self._used_tokens += tokens
        self._allocations[priority] += tokens

    def _truncate_content(self, content: str, token_budget: int, source_path: Path | None) -> str:
        """Truncate content using a provided strategy or plain truncation."""
        char_budget = self.tokens_to_characters(token_budget)
        if char_budget <= 0:
            return ""

        if source_path is not None and self.truncator is not None:
            truncated = self.truncator(
                source_path,
                char_budget,
                max_tokens=token_budget,
                limit=self.tokens_to_characters(self.model_context_limit),
            )
        else:
            truncated = self._truncate_plain(content, char_budget)

        if not truncated:
            return ""
        return truncated

    def _truncate_plain(self, content: str, limit: int) -> str:
        """Truncate plain content with marker, preferring line boundaries."""
        marker_len = len(TRUNCATION_MARKER) + 1  # +1 for newline
        if limit <= marker_len:
            return ""

        available = limit - marker_len
        truncated = content[:available]

        last_newline = truncated.rfind("\n")
        if last_newline > available // 2:
            truncated = truncated[:last_newline]

        return f"{truncated}\n{TRUNCATION_MARKER}"

    def summary(self) -> dict[str, int]:
        """Return allocation summary by priority (tokens)."""
        return {f"priority_{p}": tokens for p, tokens in self._allocations.items()}

    def allocate_with_dependencies(
        self,
        priority: Priority,
        content: str,
        target_symbol: str,
        source_path: Path | None = None,
    ) -> str:
        """Allocate content with AST-aware dependency slicing.

        Prioritizes the target symbol and its dependencies, then fills
        remaining budget with related code.

        Args:
            priority: Priority level for allocation
            content: Full source code content
            target_symbol: Name of the primary symbol to include
            source_path: Optional path for syntax-aware truncation

        Returns:
            Sliced content fitting within token budget
        """
        if not content or not target_symbol:
            return self.allocate(priority, content, source_path)

        token_budget = self.remaining()
        if token_budget <= 0:
            return ""

        # Try to use DependencyWalker for semantic slicing
        try:
            from jpscripts.core.dependency_walker import DependencyWalker

            walker = DependencyWalker(content)
            sliced = walker.slice_to_budget(target_symbol, token_budget)

            if sliced:
                return self.allocate(priority, sliced, source_path)
        except ImportError:
            logger.debug("DependencyWalker not available, using basic allocation")
        except Exception as exc:
            logger.debug("Semantic slicing failed: %s", exc)

        # Fall back to basic allocation
        return self.allocate(priority, content, source_path)


class SemanticSlicer:
    """Semantic code slicer using AST analysis.

    Provides higher-level interface for slicing code based on
    symbol relationships and token budgets.

    [invariant:typing] All types explicit; mypy --strict compliant
    """

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        model: str = "gpt-4o",
    ) -> None:
        """Initialize the semantic slicer.

        Args:
            token_counter: Optional token counter (creates default if None)
            model: Model name for token counting
        """
        self._token_counter = token_counter or TokenCounter()
        self._model = model

    def slice_for_context(
        self,
        source: str,
        target_symbol: str,
        max_tokens: int,
    ) -> str:
        """Slice source code to include target and dependencies.

        Args:
            source: Full Python source code
            target_symbol: Primary symbol to include
            max_tokens: Maximum token budget

        Returns:
            Sliced code within budget
        """
        try:
            from jpscripts.core.dependency_walker import DependencyWalker

            walker = DependencyWalker(source)
            return walker.slice_to_budget(target_symbol, max_tokens)
        except ImportError:
            # Fall back to simple truncation
            max_chars = max_tokens * 4
            return source[:max_chars]

    def prioritize_files(
        self,
        files: list[Path],
        target_symbols: list[str],
        max_tokens: int,
    ) -> list[tuple[Path, str]]:
        """Prioritize and slice multiple files for context.

        Args:
            files: List of file paths to process
            target_symbols: Symbols to prioritize across files
            max_tokens: Total token budget

        Returns:
            List of (path, sliced_content) tuples
        """
        results: list[tuple[Path, str]] = []
        remaining_tokens = max_tokens

        # First pass: find files containing target symbols
        file_relevance: list[tuple[Path, int, str]] = []

        for file_path in files:
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            # Check if file contains any target symbols
            relevance = 0
            for symbol in target_symbols:
                if symbol in content:
                    relevance += 1

            file_relevance.append((file_path, relevance, content))

        # Sort by relevance (most relevant first)
        file_relevance.sort(key=lambda x: -x[1])

        # Second pass: allocate tokens
        for file_path, relevance, content in file_relevance:
            if remaining_tokens <= 0:
                break

            if relevance > 0:
                # Slice for target symbols
                for symbol in target_symbols:
                    if symbol in content:
                        sliced = self.slice_for_context(
                            content, symbol, remaining_tokens
                        )
                        if sliced:
                            token_count = self._token_counter.count_tokens(
                                sliced, self._model
                            )
                            results.append((file_path, sliced))
                            remaining_tokens -= token_count
                            break
            else:
                # Include head of file for context
                tokens_for_file = min(remaining_tokens, 500)
                max_chars = tokens_for_file * 4
                sliced = content[:max_chars]
                token_count = self._token_counter.count_tokens(sliced, self._model)
                results.append((file_path, sliced))
                remaining_tokens -= token_count

        return results
