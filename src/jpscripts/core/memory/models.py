"""Memory data models and protocol definitions.

This module contains the core dataclasses and protocols used throughout
the memory subsystem.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

from jpscripts.core.result import JPScriptsError, Result

if TYPE_CHECKING:
    from lancedb.pydantic import (
        LanceModel as LanceModelBase,  # pyright: ignore[reportMissingTypeStubs]
    )
else:  # pragma: no cover - runtime fallbacks when optional deps are missing

    class LanceModelBase:  # type: ignore[misc]
        """Fallback LanceDB model base when dependency is missing."""

        pass


# -----------------------------------------------------------------------------
# LanceDB Protocol Types
# -----------------------------------------------------------------------------


class LanceSearchProtocol(Protocol):
    def limit(self, n: int) -> LanceSearchProtocol: ...

    def to_pydantic(self, model: type[LanceModelBase]) -> Sequence[object]: ...


class PandasDataFrameProtocol(Protocol):
    def head(self, n: int) -> PandasDataFrameProtocol: ...

    def iterrows(self) -> Iterable[tuple[int, Mapping[str, object]]]: ...


class LanceTableProtocol(Protocol):
    schema: object

    def add_column(self, name: str, dtype: object) -> object: ...

    def add(self, records: Sequence[object]) -> object: ...

    def search(self, vector: Sequence[float]) -> LanceSearchProtocol: ...

    def to_pandas(self) -> PandasDataFrameProtocol: ...


class LanceDBConnectionProtocol(Protocol):
    def table_names(self) -> Sequence[str]: ...

    def create_table(
        self, name: str, schema: type[LanceModelBase], exist_ok: bool = ...
    ) -> LanceTableProtocol: ...

    def open_table(self, name: str) -> LanceTableProtocol: ...


class LanceDBModuleProtocol(Protocol):
    def connect(self, uri: str) -> LanceDBConnectionProtocol: ...


# Type alias for readability
LanceTable: TypeAlias = LanceTableProtocol


# -----------------------------------------------------------------------------
# Record Protocol Types (for LanceDB result rows)
# -----------------------------------------------------------------------------


class MemoryRecordProtocol(Protocol):
    id: str
    timestamp: str
    content: str
    tags: list[str] | None
    embedding: list[float] | None
    source_path: str | None
    related_files: list[str] | None


class PatternRecordProtocol(Protocol):
    id: str
    created_at: str
    pattern_type: str
    description: str
    trigger: str
    solution: str
    source_traces: list[str] | None
    confidence: float
    embedding: list[float] | None


# -----------------------------------------------------------------------------
# Core Data Models
# -----------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class MemoryEntry:
    """A single memory entry with content, metadata, and optional embedding."""

    id: str
    ts: str
    content: str
    tags: list[str]
    tokens: list[str]
    embedding: list[float] | None = None
    source_path: str | None = None
    content_hash: str | None = None
    related_files: list[str] = field(default_factory=list)


@dataclass
class Pattern:
    """A generalized pattern extracted from successful execution traces.

    Patterns are synthesized from clusters of similar successful fixes
    and stored in a dedicated LanceDB collection for RAG-based injection
    into agent prompts.
    """

    id: str
    created_at: str
    pattern_type: str  # "fix_pattern", "refactor_pattern", "test_pattern"
    description: str
    trigger: str  # When to apply this pattern
    solution: str  # What to do
    source_traces: list[str]  # Trace IDs that contributed
    confidence: float  # 0.0-1.0
    embedding: list[float] | None = None


# -----------------------------------------------------------------------------
# Store Protocols
# -----------------------------------------------------------------------------


class MemoryStore(Protocol):
    """Protocol for memory storage implementations."""

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]: ...

    def search(
        self,
        query_vec: list[float] | None,
        limit: int,
        *,
        query_tokens: list[str] | None = None,
    ) -> Result[list[MemoryEntry], JPScriptsError]: ...

    def prune(self, root: Path) -> Result[int, JPScriptsError]: ...  # noqa: F821


class EmbeddingClientProtocol(Protocol):
    """Protocol for embedding client implementations."""

    @property
    def dimension(self) -> int | None: ...

    def available(self) -> bool: ...

    def embed(self, texts: list[str]) -> list[list[float]] | None: ...


# Import Path for type annotation (at runtime only for Protocol)
from pathlib import Path as _Path  # noqa: F401

# Re-export LanceModelBase for other modules
__all__ = [
    "EmbeddingClientProtocol",
    "LanceDBConnectionProtocol",
    "LanceDBModuleProtocol",
    "LanceModelBase",
    "LanceSearchProtocol",
    "LanceTable",
    "LanceTableProtocol",
    "MemoryEntry",
    "MemoryRecordProtocol",
    "MemoryStore",
    "PandasDataFrameProtocol",
    "Pattern",
    "PatternRecordProtocol",
    "T",
]
