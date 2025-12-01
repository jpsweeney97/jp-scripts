"""Memory store implementations.

This module provides storage backends for memory entries:
- JsonlArchiver: JSONL file-based storage with keyword search
- LanceDBStore: Vector database with semantic search
- HybridMemoryStore: Combined JSONL + LanceDB with RRF ranking
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import cast
from uuid import uuid4

from jpscripts.core.config import AppConfig, ConfigError
from jpscripts.core.console import get_logger
from jpscripts.core.result import (
    CapabilityMissingError,
    ConfigurationError,
    Err,
    JPScriptsError,
    Ok,
    Result,
)

from .models import (
    LanceDBConnectionProtocol,
    LanceDBModuleProtocol,
    LanceModelBase,
    LanceTable,
    MemoryEntry,
    MemoryRecordProtocol,
    MemoryStore,
)

logger = get_logger(__name__)

# Constants
MAX_ENTRIES = 5000
DEFAULT_STORE = Path.home() / ".jp_memory.lance"
FALLBACK_SUFFIX = ".jsonl"
_TOKENIZE_PATTERN = re.compile(r"[a-z0-9]+")

# TODO: Load from a resource file to enable updates without code changes.
STOPWORDS = {
    "the",
    "and",
    "or",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "by",
    "with",
    "is",
    "it",
    "this",
    "that",
    "as",
    "at",
    "be",
    "from",
    "are",
    "we",
    "use",
    "uses",
    "using",
    "self",
    "cls",
    "def",
    "class",
    "import",
    "return",
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _resolve_store(config: AppConfig | None = None, store_path: Path | None = None) -> Path:
    """Resolve the store path from config or defaults."""
    if store_path:
        return Path(store_path).expanduser()
    if config and getattr(config, "memory_store", None):
        return Path(config.memory_store).expanduser()
    return DEFAULT_STORE


def _fallback_path(base_path: Path) -> Path:
    """Get the JSONL fallback path for a store."""
    return base_path.with_suffix(FALLBACK_SUFFIX)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into words, filtering stopwords."""
    return [t for t in _TOKENIZE_PATTERN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def _format_entry(entry: MemoryEntry) -> str:
    """Format a memory entry for display."""
    tags = f"[{', '.join(entry.tags)}]" if entry.tags else ""
    return f"{entry.ts} {tags} {entry.content}".strip()


def _compute_file_hash(path: Path) -> str | None:
    """Compute MD5 hash of file content. Returns None if file cannot be read."""
    try:
        resolved = path.resolve()
        digest = hashlib.md5()
        with resolved.open("rb") as fh:
            while True:
                chunk = fh.read(4096)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _parse_entry(raw: dict[str, object]) -> MemoryEntry:
    """Parse a raw JSON dict into a MemoryEntry."""
    content = str(raw.get("content", "")).strip()
    raw_tags = raw.get("tags", [])
    tags = (
        [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        if isinstance(raw_tags, list)
        else []
    )
    token_source = f"{content} {' '.join(tags)}".strip()
    raw_tokens = raw.get("tokens")
    if isinstance(raw_tokens, list) and raw_tokens:
        tokens = [str(tok) for tok in raw_tokens if str(tok)]
    else:
        tokens = _tokenize(token_source)
    embedding = raw.get("embedding")
    embedding_list = [float(val) for val in embedding] if isinstance(embedding, list) else None
    raw_source_path = raw.get("source_path")
    source_path = str(raw_source_path) if raw_source_path else None
    raw_content_hash = raw.get("content_hash")
    content_hash = str(raw_content_hash) if raw_content_hash else None
    raw_related = raw.get("related_files")
    related_files = (
        [str(p) for p in raw_related if str(p).strip()] if isinstance(raw_related, list) else []
    )
    return MemoryEntry(
        id=str(raw.get("id", uuid4().hex)),
        ts=str(raw.get("ts", raw.get("timestamp", ""))),
        content=content,
        tags=tags,
        tokens=tokens,
        embedding=embedding_list,
        source_path=source_path,
        content_hash=content_hash,
        related_files=related_files,
    )


def _iter_entries(path: Path) -> Iterator[MemoryEntry]:
    """Generator-based entry loading for memory efficiency.

    Yields entries one at a time without loading entire file into memory.
    """
    if not path.exists():
        return

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield _parse_entry(raw)
    except OSError as exc:
        logger.debug("Failed to read memory entries from %s: %s", path, exc)
        return


def _load_entries(path: Path, max_entries: int = MAX_ENTRIES) -> list[MemoryEntry]:
    """Load entries with limit, using generator internally for efficiency."""
    return list(itertools.islice(_iter_entries(path), max_entries))


def _append_entry(path: Path, entry: MemoryEntry) -> None:
    """Append an entry to the JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": entry.id,
        "ts": entry.ts,
        "content": entry.content,
        "tags": entry.tags,
        "tokens": entry.tokens,
        "embedding": entry.embedding,
        "source_path": entry.source_path,
        "content_hash": entry.content_hash,
        "related_files": entry.related_files,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_entries(path: Path, entries: Sequence[MemoryEntry]) -> None:
    """Atomically rewrite the JSONL file with new entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            record = {
                "id": entry.id,
                "ts": entry.ts,
                "content": entry.content,
                "tags": entry.tags,
                "tokens": entry.tokens,
                "embedding": entry.embedding,
                "source_path": entry.source_path,
                "content_hash": entry.content_hash,
                "related_files": entry.related_files,
            }
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp_path, path)


def _streaming_keyword_search(
    path: Path,
    query_tokens: list[str],
    limit: int,
    *,
    tag_filter: set[str] | None = None,
) -> list[tuple[MemoryEntry, float]]:
    """Stream entries and maintain top-k scored results using a min-heap.

    This avoids loading all entries into memory at once. Uses heapq to maintain
    only the top `limit` entries as we stream through the file.

    Args:
        path: Path to the JSONL file.
        query_tokens: Tokenized query for scoring.
        limit: Maximum number of results to return.
        tag_filter: Optional set of tags. If provided, only entries with at least
                   one matching tag are considered (pre-filter before scoring).
    """
    import heapq

    if not query_tokens:
        return []

    # Use negative scores for min-heap to get max-k behavior
    # Heap entries: (neg_score, counter, entry) - counter breaks ties
    heap: list[tuple[float, int, MemoryEntry]] = []
    counter = 0

    for entry in _iter_entries(path):
        # Pre-filter by tags if specified (metadata indexing optimization)
        if tag_filter is not None:
            entry_tags = set(entry.tags)
            if not entry_tags.intersection(tag_filter):
                continue

        score = _score(query_tokens, entry)
        if score > 0:
            counter += 1
            if len(heap) < limit:
                heapq.heappush(heap, (score, counter, entry))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, entry))

    # Extract results in descending score order
    results = [(entry, score) for score, _counter, entry in heap]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _score(query_tokens: list[str], entry: MemoryEntry) -> float:
    """Score an entry based on keyword overlap with time decay."""
    if not query_tokens or not entry.tokens:
        return 0.0

    q_counts = Counter(query_tokens)
    e_counts = Counter(entry.tokens)
    overlap = sum(min(q_counts[t], e_counts[t]) for t in set(q_counts) & set(e_counts))
    tag_overlap = len(set(entry.tags) & set(query_tokens))
    base_score = float(overlap + 0.5 * tag_overlap)
    if base_score == 0.0:
        return 0.0

    decay = 1.0
    try:
        timestamp = datetime.fromisoformat(entry.ts)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        days_since = max((datetime.now(UTC) - timestamp).days, 0)
        decay = 1 / (1 + 0.1 * float(days_since))
    except Exception:
        decay = 1.0

    return base_score * decay


# -----------------------------------------------------------------------------
# LanceDB Helpers
# -----------------------------------------------------------------------------


def _load_lancedb_dependencies() -> tuple[LanceDBModuleProtocol, type[LanceModelBase]] | None:
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


# -----------------------------------------------------------------------------
# Store Implementations
# -----------------------------------------------------------------------------


class LanceDBStore(MemoryStore):
    """Vector store implementation using LanceDB."""

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
        self._table: LanceTable | None = None
        self._embedding_dim: int | None = None

    def _ensure_table(self, embedding_dim: int) -> LanceTable:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self._embedding_dim is not None and self._embedding_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: {self._embedding_dim} != {embedding_dim}"
            )

        if self._table is None or self._embedding_dim is None:
            db: LanceDBConnectionProtocol = self._lancedb.connect(str(self._db_path))
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
                                "Unable to add related_files column to LanceDB table; proceeding without schema update."
                            )
                except Exception:
                    logger.debug(
                        "Skipping related_files schema check; proceeding with existing LanceDB schema."
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
                    tokens=_tokenize(token_source),
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


class JsonlArchiver(MemoryStore):
    """JSONL file-based memory storage with keyword search."""

    def __init__(self, path: Path, max_entries: int = MAX_ENTRIES) -> None:
        self._path = path
        self._max_entries = max_entries

    @property
    def path(self) -> Path:
        return self._path

    def load_entries(self) -> list[MemoryEntry]:
        return _load_entries(self._path, self._max_entries)

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        try:
            _append_entry(self._path, entry)
        except OSError as exc:
            return Err(
                ConfigurationError(
                    "Failed to append memory entry",
                    context={"path": str(self._path), "error": str(exc)},
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
        _ = query_vec
        if not query_tokens:
            return Ok([])
        # Use streaming search to avoid loading all entries into memory
        # Pass tag_filter for pre-filtering during streaming
        scored = _streaming_keyword_search(self._path, query_tokens, limit, tag_filter=tag_filter)
        return Ok([entry for entry, _score_val in scored])

    def prune(self, root: Path) -> Result[int, JPScriptsError]:
        entries = self.load_entries()
        if not entries:
            return Ok(0)

        workspace_root = Path(root).expanduser().resolve()
        kept: list[MemoryEntry] = []
        pruned_count = 0

        for entry in entries:
            if entry.source_path is None:
                kept.append(entry)
                continue

            source = Path(entry.source_path)
            if not source.is_absolute():
                source = workspace_root / source

            try:
                if not source.exists():
                    pruned_count += 1
                    logger.debug(
                        "Pruning stale memory entry: %s (file missing: %s)",
                        entry.id,
                        entry.source_path,
                    )
                    continue

                # Check for content drift via hash mismatch
                if entry.content_hash is not None:
                    current_hash = _compute_file_hash(source)
                    if current_hash is not None and current_hash != entry.content_hash:
                        pruned_count += 1
                        logger.debug("Pruning drifted memory entry: %s (hash mismatch)", entry.id)
                        continue

                kept.append(entry)
            except OSError:
                kept.append(entry)

        try:
            _write_entries(self._path, kept)
        except OSError as exc:
            return Err(
                ConfigurationError(
                    "Failed to rewrite pruned memory archive", context={"error": str(exc)}
                )
            )

        return Ok(pruned_count)


class HybridMemoryStore(MemoryStore):
    """Hybrid store combining JSONL archiver with LanceDB vector store.

    Uses Reciprocal Rank Fusion (RRF) to combine keyword and vector search results.
    """

    def __init__(self, archiver: JsonlArchiver, vector_store: LanceDBStore | None) -> None:
        self._archiver = archiver
        self._vector_store = vector_store

    @property
    def archiver(self) -> JsonlArchiver:
        return self._archiver

    @property
    def vector_store(self) -> LanceDBStore | None:
        return self._vector_store

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        match self._archiver.add(entry):
            case Err(err):
                return Err(err)
            case Ok(_):
                pass

        if self._vector_store and entry.embedding is not None:
            vector_result = self._vector_store.add(entry)
            if isinstance(vector_result, Err):
                return vector_result

        return Ok(entry)

    def search(
        self,
        query_vec: list[float] | None,
        limit: int,
        *,
        query_tokens: list[str] | None = None,
        tag_filter: set[str] | None = None,
    ) -> Result[list[MemoryEntry], JPScriptsError]:
        # Use streaming keyword search to avoid loading all entries
        # We fetch more than limit to allow for RRF fusion
        rrf_fetch_limit = limit * 3  # Fetch extra for better RRF coverage

        vector_results: list[MemoryEntry] = []
        vector_ranks: dict[str, int] = {}

        if self._vector_store and query_vec is not None:
            match self._vector_store.search(
                query_vec, rrf_fetch_limit, query_tokens=query_tokens, tag_filter=tag_filter
            ):
                case Err(err):
                    return Err(err)
                case Ok(results):
                    vector_results = results
                    vector_ranks = {entry.id: idx + 1 for idx, entry in enumerate(results)}

        keyword_ranks: dict[str, int] = {}
        keyword_entries: dict[str, MemoryEntry] = {}
        if query_tokens:
            # Use streaming search instead of loading all entries
            # Pass tag_filter for pre-filtering during streaming
            kw_scored = _streaming_keyword_search(
                self._archiver.path, query_tokens, rrf_fetch_limit, tag_filter=tag_filter
            )
            keyword_ranks = {entry.id: idx + 1 for idx, (entry, _score_val) in enumerate(kw_scored)}
            keyword_entries = {entry.id: entry for entry, _score_val in kw_scored}

        if not vector_ranks and not keyword_ranks:
            return Ok([])

        k_const = 60.0
        # Build entry lookup only from results we have, not all entries
        entry_lookup: dict[str, MemoryEntry] = keyword_entries.copy()
        for entry in vector_results:
            entry_lookup[entry.id] = entry

        fused: list[tuple[float, MemoryEntry]] = []
        seen_ids = set(vector_ranks) | set(keyword_ranks)
        for entry_id in seen_ids:
            v_rank = vector_ranks.get(entry_id)
            k_rank = keyword_ranks.get(entry_id)
            score = 0.0
            if v_rank is not None:
                score += 1.0 / (k_const + v_rank)
            if k_rank is not None:
                score += 1.0 / (k_const + k_rank)
            found_entry = entry_lookup.get(entry_id)
            if found_entry is not None:
                fused.append((score, found_entry))

        fused.sort(key=lambda item: item[0], reverse=True)
        return Ok([entry for _, entry in fused[:limit]])

    def prune(self, root: Path) -> Result[int, JPScriptsError]:
        return self._archiver.prune(root)


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def get_memory_store(
    config: AppConfig,
    store_path: Path | None = None,
) -> Result[MemoryStore, ConfigError | CapabilityMissingError]:
    """Get a memory store for the given configuration.

    Returns a HybridMemoryStore combining JSONL archiving with LanceDB vector search.
    """
    from .embedding import _embedding_settings

    resolved_store = _resolve_store(config, store_path)
    archiver = JsonlArchiver(_fallback_path(resolved_store))

    use_semantic, _model_name, _server_url = _embedding_settings(config)
    vector_store: LanceDBStore | None = None
    if use_semantic:
        deps = _load_lancedb_dependencies()
        if deps is None:
            return Err(
                CapabilityMissingError(
                    'LanceDB is required for semantic memory. Install with `pip install "jpscripts[ai]"`.',
                    context={"path": str(resolved_store)},
                )
            )
        lancedb_module, lance_model_base = deps
        try:
            vector_store = LanceDBStore(resolved_store, lancedb_module, lance_model_base)
        except Exception as exc:  # pragma: no cover - defensive
            return Err(
                ConfigError(f"Failed to initialize LanceDB store at {resolved_store}: {exc}")
            )

    return Ok(HybridMemoryStore(archiver, vector_store))


__all__ = [
    "DEFAULT_STORE",
    "FALLBACK_SUFFIX",
    "MAX_ENTRIES",
    "STOPWORDS",
    "HybridMemoryStore",
    "JsonlArchiver",
    "LanceDBStore",
    "_compute_file_hash",
    "_fallback_path",
    "_format_entry",
    "_iter_entries",
    "_load_entries",
    "_parse_entry",
    "_resolve_store",
    "_score",
    "_streaming_keyword_search",
    "_tokenize",
    "_write_entries",
    "get_memory_store",
]
