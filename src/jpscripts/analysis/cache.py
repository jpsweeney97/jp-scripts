"""mtime-based AST caching for dependency analysis.

This module provides:
- ASTCache: LRU cache with mtime-based invalidation for parsed ASTs
- get_default_cache(): Access the module-level shared cache

The cache stores parsed AST trees keyed by file path, with automatic
invalidation when file mtime or size changes.

[invariant:typing] All types are explicit; mypy --strict compliant.
"""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MAX_CACHE_ENTRIES: Final[int] = 100


@dataclass(frozen=True)
class _CacheKey:
    """Cache key based on file identity.

    Files are identified by path + mtime + size to detect changes.
    """

    path: str
    mtime_ns: int
    size: int


@dataclass
class _CacheEntry:
    """Cached AST with source and metadata."""

    tree: ast.Module
    source: str
    key: _CacheKey


class ASTCache:
    """mtime-based cache for parsed AST trees.

    Uses LRU eviction with configurable max entries. Entries are automatically
    invalidated when file mtime or size changes.

    Thread-safe for concurrent access.

    Usage:
        cache = ASTCache()
        result = cache.get(path)
        if result is None:
            source = path.read_text()
            tree = ast.parse(source)
            cache.put(path, tree, source)
        else:
            tree, source = result

    [invariant:typing] All types explicit; mypy --strict compliant
    """

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES) -> None:
        """Initialize the cache.

        Args:
            max_entries: Maximum entries before LRU eviction (default: 100)
        """
        self._cache: dict[str, _CacheEntry] = {}
        self._max_entries = max_entries
        self._access_order: list[str] = []
        self._lock = threading.Lock()

    def get(self, path: Path) -> tuple[ast.Module, str] | None:
        """Get cached AST if valid.

        Checks that file mtime and size haven't changed since caching.

        Args:
            path: Path to the Python file

        Returns:
            (ast.Module, source) if cached and valid, None otherwise
        """
        key = self._make_key(path)
        if key is None:
            return None

        with self._lock:
            entry = self._cache.get(str(path))
            if entry is None:
                return None

            # Check if still valid (mtime/size unchanged)
            if entry.key != key:
                del self._cache[str(path)]
                self._access_order.remove(str(path))
                return None

            # Update access order (LRU)
            self._touch_unlocked(str(path))
            return entry.tree, entry.source

    def put(self, path: Path, tree: ast.Module, source: str) -> None:
        """Store AST with current file metadata.

        Args:
            path: Path to the Python file
            tree: Parsed AST module
            source: Original source code
        """
        key = self._make_key(path)
        if key is None:
            return

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_entries:
                self._evict_lru_unlocked()

            str_path = str(path)
            entry = _CacheEntry(tree=tree, source=source, key=key)
            self._cache[str_path] = entry
            self._touch_unlocked(str_path)

    def invalidate(self, path: Path) -> None:
        """Remove entry for path.

        Args:
            path: Path to invalidate
        """
        with self._lock:
            str_path = str(path)
            if str_path in self._cache:
                del self._cache[str_path]
            if str_path in self._access_order:
                self._access_order.remove(str_path)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dict with 'entries' and 'max_entries' counts
        """
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
            }

    def _make_key(self, path: Path) -> _CacheKey | None:
        """Create cache key from file stats.

        Returns None if file cannot be stat'd.
        """
        try:
            resolved = path.resolve()
            stat = resolved.stat()
            return _CacheKey(
                path=str(resolved),
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
            )
        except OSError:
            return None

    def _touch_unlocked(self, str_path: str) -> None:
        """Update LRU access order. Must hold lock."""
        if str_path in self._access_order:
            self._access_order.remove(str_path)
        self._access_order.append(str_path)

    def _evict_lru_unlocked(self) -> None:
        """Remove least recently used entry. Must hold lock."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]


# Module-level default cache (lazy initialized)
_default_cache: ASTCache | None = None
_default_cache_lock = threading.Lock()


def get_default_cache() -> ASTCache:
    """Get the module-level shared cache.

    Creates the cache on first access (thread-safe).

    Returns:
        The shared ASTCache instance
    """
    global _default_cache
    if _default_cache is None:
        with _default_cache_lock:
            if _default_cache is None:
                _default_cache = ASTCache()
    return _default_cache


def reset_default_cache() -> None:
    """Reset the default cache (primarily for testing)."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is not None:
            _default_cache.clear()
        _default_cache = None


__all__ = [
    "MAX_CACHE_ENTRIES",
    "ASTCache",
    "get_default_cache",
    "reset_default_cache",
]
