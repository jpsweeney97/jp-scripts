"""Memory retrieval and clustering functions.

This module provides functions for:
- Clustering memories by embedding similarity
- Ranking results by file relationships (graph expansion)
- Synthesizing clusters into canonical entries
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from math import sqrt
from uuid import uuid4

from jpscripts.core.config import AppConfig
from jpscripts.core.result import (
    CapabilityMissingError,
    ConfigurationError,
    Err,
    JPScriptsError,
    Ok,
    Result,
)
from jpscripts.providers import CompletionOptions, ProviderError
from jpscripts.providers import Message as ProviderMessage
from jpscripts.providers.factory import get_provider

from .embedding import EmbeddingClient, _compose_embedding_text, _embedding_settings
from .models import MemoryEntry
from .store import HybridMemoryStore, _tokenize, get_memory_store


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _graph_expand(entries: Sequence[MemoryEntry]) -> list[MemoryEntry]:
    """Re-rank entries by file relationship graph.

    Boosts entries that share files with the top result or are related
    through the file dependency graph.
    """
    if not entries:
        return []

    top = entries[0]
    top_related = set(top.related_files)
    related_union: set[str] = set()
    for entry in entries:
        related_union.update(entry.related_files)

    def _rank_score(index: int, entry: MemoryEntry) -> float:
        base = 1.0 / float(index + 1)
        entry_related = set(entry.related_files)
        score = base
        if top_related and entry_related & top_related:
            score += 0.75
        if top.source_path and entry.source_path and top.source_path == entry.source_path:
            score += 0.5
        if top.source_path and top.source_path in entry_related:
            score += 0.5
        if entry.source_path and entry.source_path in related_union:
            score += 0.25
        return score

    ranked = sorted(
        enumerate(entries),
        key=lambda item: (_rank_score(item[0], item[1]), -item[0]),
        reverse=True,
    )
    return [entries[idx] for idx, _ in ranked]


def _vectorized_cluster(
    candidates: list[MemoryEntry],
    similarity_threshold: float,
) -> list[list[MemoryEntry]]:
    """Cluster entries using vectorized numpy operations.

    Uses Union-Find with batch similarity computation for O(n²) matrix ops
    instead of O(n²) Python loops. The numpy matrix multiplication is
    highly optimized and runs much faster for large datasets.
    """
    try:
        import numpy as np
    except ImportError:
        # Fallback to simple clustering if numpy unavailable
        return _simple_cluster(candidates, similarity_threshold)

    n = len(candidates)
    if n == 0:
        return []

    # Build embedding matrix (n x d)
    embeddings = []
    for entry in candidates:
        if entry.embedding:
            embeddings.append(entry.embedding)
        else:
            embeddings.append([])

    # Filter to only entries with embeddings
    valid_indices = [i for i, e in enumerate(embeddings) if len(e) > 0]
    if not valid_indices:
        return []

    # Extract valid embeddings into numpy array
    valid_embeddings = np.array([embeddings[i] for i in valid_indices], dtype=np.float32)

    # Normalize for cosine similarity: sim(a,b) = (a·b) / (|a||b|)
    norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = valid_embeddings / norms

    # Compute full similarity matrix in one operation: O(n²) but vectorized
    similarity_matrix = normalized @ normalized.T

    # Union-Find clustering
    parent = list(range(len(valid_indices)))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union entries that are similar
    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)

    # Group by cluster root
    cluster_map: dict[int, list[int]] = {}
    for i in range(len(valid_indices)):
        root = find(i)
        if root not in cluster_map:
            cluster_map[root] = []
        cluster_map[root].append(valid_indices[i])

    # Build result clusters (only those with >1 entry)
    clusters = []
    for indices in cluster_map.values():
        if len(indices) > 1:
            cluster = [candidates[i] for i in sorted(indices, key=lambda i: candidates[i].ts)]
            clusters.append(cluster)

    return clusters


def _simple_cluster(
    candidates: list[MemoryEntry],
    similarity_threshold: float,
) -> list[list[MemoryEntry]]:
    """Fallback clustering when numpy is unavailable.

    O(n²) in Python loops - slower but works without numpy.
    """
    clusters: list[list[MemoryEntry]] = []

    for entry in candidates:
        embedding = entry.embedding
        if embedding is None:
            continue
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            rep_embedding = representative.embedding or []
            if _cosine_similarity(embedding, rep_embedding) >= similarity_threshold:
                cluster.append(entry)
                placed = True
                break
        if not placed:
            clusters.append([entry])

    return [cluster for cluster in clusters if len(cluster) > 1]


async def cluster_memories(
    config: AppConfig,
    similarity_threshold: float = 0.85,
) -> Result[list[list[MemoryEntry]], JPScriptsError]:
    """Group memories into clusters based on cosine similarity.

    Returns clusters where each cluster contains entries with embeddings
    similar to the cluster representative (first entry).

    Uses vectorized numpy operations for O(n²) matrix computation instead
    of O(n²) Python loops, providing significant speedup for large datasets.
    """
    match get_memory_store(config):
        case Err(err):
            if isinstance(err, JPScriptsError):
                return Err(err)
            return Err(ConfigurationError(str(err)))
        case Ok(store):
            pass

    if not isinstance(store, HybridMemoryStore) or store.vector_store is None:
        return Err(CapabilityMissingError("LanceDB is required for clustering memories."))

    entries = await asyncio.to_thread(store.archiver.load_entries)
    candidates = [entry for entry in entries if entry.embedding is not None]
    if not candidates:
        return Ok([])

    candidates.sort(key=lambda e: e.ts)

    # Use vectorized clustering for performance
    dense_clusters = await asyncio.to_thread(_vectorized_cluster, candidates, similarity_threshold)
    return Ok(dense_clusters)


async def synthesize_cluster(
    entries: Sequence[MemoryEntry],
    config: AppConfig,
    *,
    model: str | None = None,
) -> Result[MemoryEntry, JPScriptsError]:
    """Synthesize a canonical memory from a cluster.

    Uses an LLM to merge similar memories into a single, authoritative entry.
    """
    if not entries:
        return Err(ConfigurationError("Cannot synthesize from an empty cluster."))

    model_id = model or config.default_model
    try:
        provider = get_provider(config, model_id=model_id)
    except Exception as exc:
        return Err(ConfigurationError("Failed to initialize provider", context={"error": str(exc)}))

    ordered = sorted(entries, key=lambda e: e.ts)
    lines = []
    for entry in ordered:
        line_tags = ",".join(entry.tags) if entry.tags else "none"
        source = entry.source_path or "unknown"
        lines.append(f"- ts={entry.ts}; tags={line_tags}; source={source}; content={entry.content}")

    system_prompt = (
        "These memories describe the same topic. Merge them into a single, canonical 'Truth' entry. "
        "Resolve contradictions by favoring the most recent timestamp."
    )
    messages = [
        ProviderMessage(role="system", content=system_prompt),
        ProviderMessage(role="user", content="\n".join(lines)),
    ]
    options = CompletionOptions(temperature=0.2, reasoning_effort="high")

    try:
        response = await provider.complete(messages=messages, model=model_id, options=options)
    except ProviderError as exc:
        return Err(ConfigurationError("LLM synthesis failed", context={"error": str(exc)}))
    except Exception as exc:  # pragma: no cover - defensive
        return Err(ConfigurationError("LLM synthesis failed", context={"error": str(exc)}))

    synthesized_content = response.content.strip()
    aggregated_tags = sorted({tag for entry in entries for tag in entry.tags} | {"truth"})
    token_source = f"{synthesized_content} {' '.join(aggregated_tags)}".strip()
    source_paths = sorted({entry.source_path for entry in entries if entry.source_path})
    source_metadata = ";".join(source_paths) if source_paths else None

    new_entry = MemoryEntry(
        id=uuid4().hex,
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        content=synthesized_content,
        tags=aggregated_tags,
        tokens=_tokenize(token_source),
        source_path=source_metadata,
        content_hash=None,
    )

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    vectors = embedding_client.embed([_compose_embedding_text(new_entry)]) if use_semantic else None
    if vectors:
        new_entry.embedding = vectors[0]

    return Ok(new_entry)


__all__ = [
    "_cosine_similarity",
    "_graph_expand",
    "_simple_cluster",
    "_vectorized_cluster",
    "cluster_memories",
    "synthesize_cluster",
]
