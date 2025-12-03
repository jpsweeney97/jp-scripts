"""Memory store package with pluggable backends.

This package provides storage backends for memory entries:
- JsonlArchiver: JSONL file-based storage with keyword search
- LanceDBStore: Vector database with semantic search
- HybridMemoryStore: Combined JSONL + LanceDB with RRF ranking

The package uses a StorageMode enum to select backends and provides
a factory function get_memory_store() for creating configured stores.
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
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from jpscripts.core.console import get_logger

if TYPE_CHECKING:
    from jpscripts.memory.models import MemoryEntry

logger = get_logger(__name__)

# Constants
MAX_ENTRIES = 5000
DEFAULT_STORE = Path.home() / ".jp_memory.lance"
FALLBACK_SUFFIX = ".jsonl"
_TOKENIZE_PATTERN = re.compile(r"[a-z0-9]+")

# TODO: Load from a resource file to enable updates without code changes.
STOPWORDS = frozenset(
    {
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
)


class StorageMode(Enum):
    """Storage backend selection mode.

    Attributes:
        JSONL_ONLY: Use only JSONL file-based storage (keyword search).
        LANCE_ONLY: Use only LanceDB vector storage (semantic search).
        HYBRID: Use both JSONL and LanceDB with RRF fusion (default).
        NO_OP: No-operation store for testing/disabled memory.
    """

    JSONL_ONLY = auto()
    LANCE_ONLY = auto()
    HYBRID = auto()
    NO_OP = auto()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def resolve_store_path(config: object | None = None, store_path: Path | None = None) -> Path:
    """Resolve the store path from config or defaults.

    Args:
        config: Optional AppConfig with user.memory_store path.
        store_path: Optional explicit store path override.

    Returns:
        Resolved Path for the memory store.
    """
    if store_path:
        return Path(store_path).expanduser()
    if config and hasattr(config, "user"):
        user_config = getattr(config, "user", None)
        if user_config and getattr(user_config, "memory_store", None):
            return Path(user_config.memory_store).expanduser()
    return DEFAULT_STORE


def fallback_path(base_path: Path) -> Path:
    """Get the JSONL fallback path for a store."""
    return base_path.with_suffix(FALLBACK_SUFFIX)


def tokenize(text: str) -> list[str]:
    """Tokenize text into words, filtering stopwords."""
    return [t for t in _TOKENIZE_PATTERN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def format_entry(entry: MemoryEntry) -> str:
    """Format a memory entry for display."""
    tags = f"[{', '.join(entry.tags)}]" if entry.tags else ""
    return f"{entry.ts} {tags} {entry.content}".strip()


def compute_file_hash(path: Path) -> str | None:
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


def parse_entry(raw: dict[str, object]) -> MemoryEntry:
    """Parse a raw JSON dict into a MemoryEntry."""
    from jpscripts.memory.models import MemoryEntry

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
        tokens = tokenize(token_source)
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


def iter_entries(path: Path) -> Iterator[MemoryEntry]:
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
                yield parse_entry(raw)
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("Failed to read memory entries from %s: %s", path, exc)
        return


def load_entries(path: Path, max_entries: int = MAX_ENTRIES) -> list[MemoryEntry]:
    """Load entries with limit, using generator internally for efficiency."""
    return list(itertools.islice(iter_entries(path), max_entries))


def append_entry(path: Path, entry: MemoryEntry) -> None:
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


def write_entries(path: Path, entries: Sequence[MemoryEntry]) -> None:
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


def score_entry(query_tokens: list[str], entry: MemoryEntry) -> float:
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


def streaming_keyword_search(
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

    for entry in iter_entries(path):
        # Pre-filter by tags if specified (metadata indexing optimization)
        if tag_filter is not None:
            entry_tags = set(entry.tags)
            if not entry_tags.intersection(tag_filter):
                continue

        score = score_entry(query_tokens, entry)
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


# Backward-compatible aliases (underscore-prefixed)
_resolve_store = resolve_store_path
_fallback_path = fallback_path
_tokenize = tokenize
_format_entry = format_entry
_compute_file_hash = compute_file_hash
_parse_entry = parse_entry
_iter_entries = iter_entries
_load_entries = load_entries
_write_entries = write_entries
_score = score_entry
_streaming_keyword_search = streaming_keyword_search


# Re-export store implementations
from jpscripts.memory.store.hybrid import HybridMemoryStore, NoOpMemoryStore, get_memory_store
from jpscripts.memory.store.jsonl import JsonlArchiver
from jpscripts.memory.store.lance import (
    LanceDBStore,
    _load_lancedb_dependencies,
    load_lancedb_dependencies,
)

__all__ = [
    # Constants
    "DEFAULT_STORE",
    "FALLBACK_SUFFIX",
    "MAX_ENTRIES",
    "STOPWORDS",
    # Store classes
    "HybridMemoryStore",
    "JsonlArchiver",
    "LanceDBStore",
    "NoOpMemoryStore",
    # Enum
    "StorageMode",
    "_compute_file_hash",
    "_fallback_path",
    "_format_entry",
    "_iter_entries",
    "_load_entries",
    "_load_lancedb_dependencies",
    "_parse_entry",
    # Backward-compatible aliases
    "_resolve_store",
    "_score",
    "_streaming_keyword_search",
    "_tokenize",
    "_write_entries",
    "append_entry",
    "compute_file_hash",
    "fallback_path",
    "format_entry",
    # Factory
    "get_memory_store",
    "iter_entries",
    "load_entries",
    # LanceDB helpers
    "load_lancedb_dependencies",
    "parse_entry",
    # Public utility functions
    "resolve_store_path",
    "score_entry",
    "streaming_keyword_search",
    "tokenize",
    "write_entries",
]
