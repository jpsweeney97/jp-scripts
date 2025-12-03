"""Hybrid memory store combining JSONL and LanceDB backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jpscripts.core.config import AppConfig, ConfigError
from jpscripts.core.result import CapabilityMissingError, Err, JPScriptsError, Ok, Result
from jpscripts.memory.models import MemoryEntry, MemoryStore

from . import (
    fallback_path,
    resolve_store_path,
    streaming_keyword_search,
)
from .jsonl import JsonlArchiver
from .lance import LanceDBStore, load_lancedb_dependencies

if TYPE_CHECKING:
    pass


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
            kw_scored = streaming_keyword_search(
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


def get_memory_store(
    config: AppConfig,
    store_path: Path | None = None,
) -> Result[MemoryStore, ConfigError | CapabilityMissingError]:
    """Get a memory store for the given configuration.

    Returns a HybridMemoryStore combining JSONL archiving with LanceDB vector search.
    """
    from jpscripts.memory.embedding import _embedding_settings

    resolved_store = resolve_store_path(config, store_path)
    archiver = JsonlArchiver(fallback_path(resolved_store))

    use_semantic, _model_name, _server_url = _embedding_settings(config)
    vector_store: LanceDBStore | None = None
    if use_semantic:
        deps = load_lancedb_dependencies()
        if deps is None:
            return Err(
                CapabilityMissingError(
                    "LanceDB is required for semantic memory. "
                    'Install with `pip install "jpscripts[ai]"`.',
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
    "HybridMemoryStore",
    "get_memory_store",
]
