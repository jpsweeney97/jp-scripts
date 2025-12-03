"""JSONL file-based memory storage with keyword search."""

from __future__ import annotations

from pathlib import Path

from jpscripts.core.console import get_logger
from jpscripts.core.result import ConfigurationError, Err, JPScriptsError, Ok, Result
from jpscripts.memory.models import MemoryEntry, MemoryStore

from . import (
    MAX_ENTRIES,
    append_entry,
    compute_file_hash,
    load_entries,
    streaming_keyword_search,
    write_entries,
)

logger = get_logger(__name__)


class JsonlArchiver(MemoryStore):
    """JSONL file-based memory storage with keyword search."""

    def __init__(self, path: Path, max_entries: int = MAX_ENTRIES) -> None:
        self._path = path
        self._max_entries = max_entries

    @property
    def path(self) -> Path:
        return self._path

    def load_entries(self) -> list[MemoryEntry]:
        return load_entries(self._path, self._max_entries)

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        try:
            append_entry(self._path, entry)
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
        scored = streaming_keyword_search(self._path, query_tokens, limit, tag_filter=tag_filter)
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
                    current_hash = compute_file_hash(source)
                    if current_hash is not None and current_hash != entry.content_hash:
                        pruned_count += 1
                        logger.debug("Pruning drifted memory entry: %s (hash mismatch)", entry.id)
                        continue

                kept.append(entry)
            except OSError:
                kept.append(entry)

        try:
            write_entries(self._path, kept)
        except OSError as exc:
            return Err(
                ConfigurationError(
                    "Failed to rewrite pruned memory archive", context={"error": str(exc)}
                )
            )

        return Ok(pruned_count)


__all__ = ["JsonlArchiver"]
