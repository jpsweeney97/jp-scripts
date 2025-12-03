"""Memory subsystem for jpscripts.

This package provides memory storage and retrieval capabilities including:
- Semantic search using embeddings
- Keyword-based search with time decay
- Pattern extraction from execution traces
- Memory clustering and synthesis

Public API:
- save_memory: Persist a memory entry
- query_memory: Retrieve relevant memories
- prune_memory: Remove stale entries
- reindex_memory: Migrate and refresh embeddings
- cluster_memories: Group memories by similarity
- synthesize_cluster: Merge similar memories
- consolidate_patterns: Extract patterns from traces
- fetch_relevant_patterns: Retrieve patterns for prompts
- format_patterns_for_prompt: Format patterns for injection
"""

# Models
# Public API
from .api import (
    prune_memory,
    query_memory,
    reindex_memory,
    save_memory,
)

# Embedding
from .embedding import (
    EmbeddingClient,
    _compose_embedding_text,
    _embedding_settings,
    _GlobalEmbeddingClient,
    _warn_semantic_unavailable,
)
from .models import (
    EmbeddingClientProtocol,
    LanceDBConnectionProtocol,
    LanceDBModuleProtocol,
    LanceModelBase,
    LanceSearchProtocol,
    LanceTable,
    LanceTableProtocol,
    MemoryEntry,
    MemoryRecordProtocol,
    MemoryStore,
    PandasDataFrameProtocol,
    Pattern,
    PatternRecordProtocol,
)

# Patterns
from .patterns import (
    PatternStore,
    consolidate_patterns,
    fetch_relevant_patterns,
    format_patterns_for_prompt,
    get_pattern_store,
)

# Retrieval
from .retrieval import (
    _cosine_similarity,
    _graph_expand,
    cluster_memories,
    synthesize_cluster,
)

# Store
from .store import (
    DEFAULT_STORE,
    FALLBACK_SUFFIX,
    MAX_ENTRIES,
    STOPWORDS,
    HybridMemoryStore,
    JsonlArchiver,
    LanceDBStore,
    StorageMode,
    # Private but needed by some importers/tests
    _compute_file_hash,
    _fallback_path,
    _format_entry,
    _iter_entries,
    _load_entries,
    _load_lancedb_dependencies,
    _parse_entry,
    _resolve_store,
    _score,
    _streaming_keyword_search,
    _tokenize,
    _write_entries,
    get_memory_store,
)

__all__ = [
    # Store
    "DEFAULT_STORE",
    "FALLBACK_SUFFIX",
    "MAX_ENTRIES",
    "STOPWORDS",
    # Embedding
    "EmbeddingClient",
    # Models
    "EmbeddingClientProtocol",
    "HybridMemoryStore",
    "JsonlArchiver",
    "LanceDBConnectionProtocol",
    "LanceDBModuleProtocol",
    "LanceDBStore",
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
    # Patterns
    "PatternStore",
    "StorageMode",
    "_GlobalEmbeddingClient",
    "_compose_embedding_text",
    # Store - private (needed for tests/compatibility)
    "_compute_file_hash",
    "_cosine_similarity",
    "_embedding_settings",
    "_fallback_path",
    "_format_entry",
    "_graph_expand",
    "_iter_entries",
    "_load_entries",
    "_load_lancedb_dependencies",
    "_parse_entry",
    "_resolve_store",
    "_score",
    "_streaming_keyword_search",
    "_tokenize",
    "_warn_semantic_unavailable",
    "_write_entries",
    # Retrieval
    "cluster_memories",
    "consolidate_patterns",
    "fetch_relevant_patterns",
    "format_patterns_for_prompt",
    "get_memory_store",
    "get_pattern_store",
    # Public API
    "prune_memory",
    "query_memory",
    "reindex_memory",
    "save_memory",
    "synthesize_cluster",
]
