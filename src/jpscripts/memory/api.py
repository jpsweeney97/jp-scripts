"""Public API for memory operations.

This module provides the main entry points for memory operations:
- save_memory: Persist a memory entry
- query_memory: Retrieve relevant memories
- prune_memory: Remove stale entries
- reindex_memory: Migrate and refresh embeddings
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from jpscripts.core import structure
from jpscripts.core.config import AppConfig, ConfigError
from jpscripts.core.result import Err, Ok

from .embedding import EmbeddingClient, _compose_embedding_text, _embedding_settings
from .models import MemoryEntry
from .retrieval import _graph_expand
from .store import (
    HybridMemoryStore,
    _compute_file_hash,
    _fallback_path,
    _format_entry,
    _load_entries,
    _resolve_store,
    _tokenize,
    _write_entries,
    get_memory_store,
)

# -----------------------------------------------------------------------------
# File Relationship Helpers
# -----------------------------------------------------------------------------


def _normalize_related_path(path: Path, root: Path) -> str:
    """Normalize a file path relative to the workspace root."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def _is_ignored_path(path: Path, root: Path, ignore_dirs: Sequence[str]) -> bool:
    """Check if a path should be ignored based on configuration."""
    try:
        rel_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return True
    ignore_set = {ignore.strip("/").strip() for ignore in ignore_dirs if ignore.strip()}
    return any(part in ignore_set for part in rel_parts)


def _collect_related_files(source_path: Path, root: Path, ignore_dirs: Sequence[str]) -> list[str]:
    """Collect files related to the source through import dependencies."""
    if not source_path.exists():
        return []

    resolved_root = root.resolve()
    resolved_source = source_path.resolve()
    related: set[str] = set()

    for dep in structure.get_import_dependencies(resolved_source, resolved_root):
        if _is_ignored_path(dep, resolved_root, ignore_dirs):
            continue
        related.add(_normalize_related_path(dep, resolved_root))

    try:
        candidates = list(resolved_root.rglob("*.py"))
    except OSError:
        candidates = []

    for candidate in candidates:
        if candidate.resolve() == resolved_source:
            continue
        if _is_ignored_path(candidate, resolved_root, ignore_dirs):
            continue
        dependencies = structure.get_import_dependencies(candidate, resolved_root)
        if any(dep.resolve() == resolved_source for dep in dependencies):
            related.add(_normalize_related_path(candidate, resolved_root))

    return sorted(related)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def save_memory(
    content: str,
    tags: Sequence[str] | None = None,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
    source_path: str | None = None,
) -> MemoryEntry:
    """Persist a memory entry for later recall.

    Args:
        content: Memory content or ADR/lesson learned.
        tags: Tags to associate with the entry.
        config: Application configuration.
        store_path: Override store location.
        source_path: Optional file path this memory is related to.

    Returns:
        The created MemoryEntry.

    Raises:
        ConfigError: If config is not provided.
    """
    if config is None:
        raise ConfigError("AppConfig is required to save memory.")

    resolved_store = _resolve_store(config, store_path)
    root = Path(getattr(config, "workspace_root", Path.cwd())).expanduser().resolve()

    normalized_tags = [t.strip() for t in (tags or []) if t.strip()]
    content_text = content.strip()
    token_source = f"{content_text} {' '.join(normalized_tags)}".strip()

    # Compute content hash if source_path provided
    computed_hash: str | None = None
    related_files: list[str] = []
    normalized_source: str | None = None
    if source_path:
        source = Path(source_path).expanduser()
        if not source.is_absolute():
            source = root / source
        computed_hash = _compute_file_hash(source)
        normalized_source = _normalize_related_path(source, root)
        related_files = _collect_related_files(source, root, getattr(config, "ignore_dirs", []))

    entry = MemoryEntry(
        id=uuid4().hex,
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        content=content_text,
        tags=normalized_tags,
        tokens=_tokenize(token_source),
        source_path=normalized_source,
        content_hash=computed_hash,
        related_files=related_files,
    )

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    vectors = embedding_client.embed([_compose_embedding_text(entry)]) if use_semantic else None
    if vectors:
        entry.embedding = vectors[0]

    match get_memory_store(config, store_path=resolved_store):
        case Err(err):
            raise err
        case Ok(store):
            add_result = store.add(entry)
            if isinstance(add_result, Err):
                raise add_result.error

    return entry


def query_memory(
    query: str,
    limit: int = 5,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
) -> list[str]:
    """Retrieve the most relevant memory snippets for a query using reciprocal rank fusion.

    Args:
        query: Search query text.
        limit: Maximum number of results to return.
        config: Application configuration.
        store_path: Override store location.

    Returns:
        List of formatted memory entries.

    Raises:
        ConfigError: If config is not provided.
    """
    if config is None:
        raise ConfigError("AppConfig is required to query memory.")

    match get_memory_store(config, store_path=store_path):
        case Err(err):
            raise err
        case Ok(store):
            use_semantic, model_name, server_url = _embedding_settings(config)
            embedding_client = EmbeddingClient(
                model_name, enabled=use_semantic, server_url=server_url
            )
            query_vecs = embedding_client.embed([query]) if use_semantic else None
            query_vec = query_vecs[0] if query_vecs else None
            tokens = _tokenize(query)

            search_result = store.search(query_vec, limit, query_tokens=tokens)
            if isinstance(search_result, Err):
                raise search_result.error
            entries = search_result.value
            if not entries:
                return []
            ranked_entries = _graph_expand(entries)
            return [_format_entry(entry) for entry in ranked_entries[:limit]]


def reindex_memory(
    *,
    config: AppConfig | None = None,
    legacy_path: Path | None = None,
    target_path: Path | None = None,
) -> Path:
    """
    Migrate existing JSONL memory data to the LanceDB store and refresh embeddings.

    Args:
        config: Application configuration.
        legacy_path: Source JSONL file to migrate.
        target_path: Target store location.

    Returns:
        Path to the target store.

    Raises:
        ConfigError: If config is not provided.
    """
    if config is None:
        raise ConfigError("AppConfig is required to reindex memory.")

    target_store = _resolve_store(config, target_path)
    fallback_target = _fallback_path(target_store)
    source = legacy_path or fallback_target
    entries = _load_entries(source)
    if not entries:
        return target_store

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    for entry in entries:
        if entry.embedding is None and use_semantic:
            vectors = embedding_client.embed([_compose_embedding_text(entry)])
            if vectors:
                entry.embedding = vectors[0]

    _write_entries(fallback_target, entries)

    match get_memory_store(config, store_path=target_store):
        case Err(err):
            raise err
        case Ok(store):
            if isinstance(store, HybridMemoryStore) and store.vector_store:
                for entry in entries:
                    if entry.embedding:
                        _ = store.vector_store.add(entry)

    return target_store


def prune_memory(config: AppConfig) -> int:
    """Remove memory entries related to deleted files to maintain vector store hygiene.

    Loads all entries, checks if source_path exists (relative to workspace_root or absolute),
    and removes stale entries from JSONL. Triggers reindex afterward.

    Args:
        config: Application configuration with workspace_root.

    Returns:
        Count of pruned entries.
    """
    match get_memory_store(config):
        case Err(err):
            raise err
        case Ok(store):
            result = store.prune(Path(config.user.workspace_root))
            if isinstance(result, Err):
                raise result.error
            return result.value


__all__ = [
    "prune_memory",
    "query_memory",
    "reindex_memory",
    "save_memory",
]
