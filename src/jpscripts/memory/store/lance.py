"""LanceDB vector store implementation for semantic memory search."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

from jpscripts.core.console import get_logger
from jpscripts.core.result import ConfigurationError, Err, JPScriptsError, Ok, Result
from jpscripts.memory.models import (
    LanceDBConnectionProtocol,
    LanceDBModuleProtocol,
    LanceModelBase,
    LanceTable,
    MemoryEntry,
    MemoryRecordProtocol,
    MemoryStore,
)

from . import tokenize

logger = get_logger(__name__)


def load_lancedb_dependencies() -> tuple[LanceDBModuleProtocol, type[LanceModelBase]] | None:
    """Load LanceDB dependencies, returning None if unavailable."""
    try:
        lancedb = import_module("lancedb")
        pydantic_module = import_module("lancedb.pydantic")
        lance_model = cast(type[LanceModelBase], pydantic_module.LanceModel)
    except Exception as exc:
        logger.debug("LanceDB unavailable: %s", exc)
        return None
    return cast(LanceDBModuleProtocol, lancedb), lance_model


def _build_memory_record_model(base: type[LanceModelBase]) -> type[LanceModelBase]:
    """Build LanceDB model for memory records."""

    class MemoryRecord(base):  # type: ignore[misc]
        id: str
        timestamp: str
        content: str
        tags: list[str]
        embedding: list[float] | None
        source_path: str | None
        related_files: list[str] | None

    return MemoryRecord


class LanceDBStore(MemoryStore):
    """Vector store implementation using LanceDB.

    This class caches both the database connection and table to avoid
    reconnection overhead on every operation.
    """

    def __init__(
        self,
        db_path: Path,
        lancedb_module: LanceDBModuleProtocol,
        lance_model_base: type[LanceModelBase],
    ) -> None:
        self._db_path = db_path.expanduser()
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._lancedb: LanceDBModuleProtocol = lancedb_module
        self._model_cls: type[LanceModelBase] = _build_memory_record_model(lance_model_base)
        self._connection: LanceDBConnectionProtocol | None = None
        self._table: LanceTable | None = None
        self._embedding_dim: int | None = None

    def _get_connection(self) -> LanceDBConnectionProtocol:
        """Get or create a cached database connection."""
        if self._connection is None:
            self._connection = self._lancedb.connect(str(self._db_path))
        return self._connection

    def _ensure_table(self, embedding_dim: int) -> LanceTable:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self._embedding_dim is not None and self._embedding_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: {self._embedding_dim} != {embedding_dim}"
            )

        if self._table is None or self._embedding_dim is None:
            db = self._get_connection()
            model = self._model_cls
            if "memory" not in db.table_names():
                self._table = db.create_table("memory", schema=model, exist_ok=True)
            else:
                self._table = db.open_table("memory")
                try:
                    schema = getattr(self._table, "schema", None)
                    names_attr = getattr(schema, "names", None)
                    existing_names = (
                        {str(name) for name in names_attr}
                        if isinstance(names_attr, (list, tuple, set))
                        else set()
                    )
                    if "related_files" not in existing_names:
                        try:
                            self._table.add_column("related_files", list[str])
                        except Exception:
                            logger.debug(
                                "Unable to add related_files column to LanceDB table; "
                                "proceeding without schema update."
                            )
                except Exception:
                    logger.debug(
                        "Skipping related_files schema check; "
                        "proceeding with existing LanceDB schema."
                    )
            self._embedding_dim = embedding_dim
        return self._table

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        if entry.embedding is None:
            return Err(
                ConfigurationError(
                    "Embedding required for LanceDB insert", context={"id": entry.id}
                )
            )

        try:
            table = self._ensure_table(len(entry.embedding))
            model = self._model_cls(  # pyright: ignore[reportCallIssue]
                id=entry.id,
                timestamp=entry.ts,
                content=entry.content,
                tags=entry.tags,
                embedding=entry.embedding,
                source_path=entry.source_path,
                related_files=entry.related_files or None,
            )
            table.add([model])
        except Exception as exc:  # pragma: no cover - defensive
            return Err(
                ConfigurationError(
                    "Failed to persist memory to LanceDB", context={"error": str(exc)}
                )
            )

        return Ok(entry)

    def search(
        self,
        query_vec: list[float] | None,
        limit: int,
        *,
        query_tokens: list[str] | None = None,
        tag_filter: set[str] | None = None,
    ) -> Result[list[MemoryEntry], JPScriptsError]:
        if query_vec is None:
            return Ok([])

        try:
            table = self._ensure_table(len(query_vec))
        except Exception as exc:
            return Err(
                ConfigurationError("Failed to prepare LanceDB table", context={"error": str(exc)})
            )

        # Fetch extra if we need to filter by tags (post-filter for LanceDB)
        fetch_limit = limit * 3 if tag_filter is not None else limit

        try:
            results = cast(
                Sequence[MemoryRecordProtocol],
                table.search(query_vec).limit(fetch_limit).to_pydantic(self._model_cls),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return Err(ConfigurationError("LanceDB search failed", context={"error": str(exc)}))

        matches: list[MemoryEntry] = []
        for row in results:
            tags = list(row.tags or [])

            # Apply tag filter if specified
            if tag_filter is not None and not set(tags).intersection(tag_filter):
                continue

            token_source = f"{row.content} {' '.join(tags)}".strip()
            matches.append(
                MemoryEntry(
                    id=row.id,
                    ts=row.timestamp,
                    content=row.content,
                    tags=tags,
                    tokens=tokenize(token_source),
                    embedding=list(row.embedding) if row.embedding is not None else None,
                    source_path=row.source_path,
                    related_files=list(row.related_files or []),
                )
            )
            if len(matches) >= limit:
                break

        return Ok(matches)

    def prune(
        self, root: Path
    ) -> Result[int, JPScriptsError]:  # pragma: no cover - not required for LanceDB
        _ = root
        return Ok(0)


# Backward-compatible alias
_load_lancedb_dependencies = load_lancedb_dependencies


__all__ = [
    "LanceDBStore",
    "_load_lancedb_dependencies",
    "load_lancedb_dependencies",
]
